"""
TJSCC-PCT: Reusable Module Components
Contains all building blocks for the TJSCC-PCT framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# CORE UTILITIES
# ============================================================================

def index_points(points, idx):
    """Gather points according to index with bounds checking."""
    B = points.shape[0]
    N = points.shape[1]
    
    # Ensure indices are within bounds
    idx = torch.clamp(idx, 0, N - 1)
    
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz, npoint):
    """Farthest Point Sampling for downsampling."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance(src, dst):
    """Compute squared distance between all pairs of points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group points by ball query."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# ============================================================================
# INPUT EMBEDDING
# ============================================================================

class InputEmbedding(nn.Module):
    """Map input points to initial 128-dim features (Section III.A)."""
    def __init__(self, in_channel=3, out_channel=128, radius=0.1, nsample=32):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel * 2 + 3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, xyz, points=None):
        B, N, C = xyz.shape
        
        if points is None:
            points = xyz
        
        group_idx = query_ball_point(self.radius, self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - xyz.view(B, N, 1, C)
        grouped_points = index_points(points, group_idx)
        
        grouped_features = torch.cat([grouped_xyz_norm, 
                                       xyz.view(B, N, 1, -1).expand(-1, -1, self.nsample, -1),
                                       grouped_points], dim=-1)
        
        grouped_features = grouped_features.permute(0, 3, 2, 1)
        new_points = self.mlp(grouped_features)
        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        
        return xyz, new_points

# ============================================================================
# TRANSFORMER BLOCK (Multi-Head Attention)
# ============================================================================

class TransformerBlock(nn.Module):
    """Point Transformer with multi-head attention (4 heads per paper)."""
    def __init__(self, d_points, d_model, k=32, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.k = k
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        
        # CRITICAL: Ensure k doesn't exceed number of points
        k_safe = min(self.k, N)
        
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :k_safe]
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        
        q = self.w_qs(x)
        k = index_points(self.w_ks(x), knn_idx)
        v = index_points(self.w_vs(x), knn_idx)
        
        pos_enc = self.fc_delta(xyz.unsqueeze(2) - knn_xyz)
        
        attn_input = q.unsqueeze(2) - k + pos_enc
        attn = self.fc_gamma(attn_input)
        attn = F.softmax(attn / math.sqrt(self.head_dim), dim=2)
        
        weighted_v = attn * (v + pos_enc)
        res = weighted_v.sum(dim=2)
        res = self.w_out(res)
        
        return self.fc2(res) + pre, attn

# ============================================================================
# POINTNET SET ABSTRACTION (Encoder Stage)
# ============================================================================

class PointNetSetAbstraction(nn.Module):
    """Hierarchical encoder stage with downsampling."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel + 3
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Sequential(
                nn.Conv2d(last_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.npoint
        
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
        
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        
        new_points = new_points.permute(0, 3, 2, 1)
        for conv in self.mlp_convs:
            new_points = conv(new_points)
        
        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        
        return new_xyz, new_points

# ============================================================================
# POINTNET FEATURE PROPAGATION (Decoder Stage)
# ============================================================================

class PointNetFeaturePropagation(nn.Module):
    """Upsampling with skip connections."""
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        
        dists = square_distance(xyz1, xyz2)
        dist, idx = dists.topk(k=3, dim=-1, largest=False)
        dist = torch.clamp(dist, min=1e-10)
        
        weight = (1.0 / dist) / torch.sum(1.0 / dist, dim=-1, keepdim=True)
        interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points.permute(0, 2, 1)

# ============================================================================
# WIRELESS CHANNEL LAYER (Eq. 5)
# ============================================================================

class WirelessChannel(nn.Module):
    """Differentiable channel layer with AWGN and Rayleigh fading."""
    def __init__(self, mode='awgn'):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, snr_db):
        B, C = x.shape
        device = x.device
        
        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(snr_db, dtype=torch.float32, device=device)
        snr_db = snr_db.view(-1, 1)
        snr_lin = 10 ** (snr_db / 10.0)
        
        if self.mode == 'rayleigh':
            h_real = torch.randn(B, C, device=device) * math.sqrt(0.5)
            h_imag = torch.randn(B, C, device=device) * math.sqrt(0.5)
            h = torch.sqrt(h_real ** 2 + h_imag ** 2)
            noise_std = torch.sqrt(1.0 / (2.0 * snr_lin))
            noise = torch.randn_like(x) * noise_std
            return x * h + noise
        else:
            noise_std = torch.sqrt(1.0 / snr_lin)
            noise = torch.randn_like(x) * noise_std
            return x + noise

# ============================================================================
# POWER NORMALIZATION (Eq. 4)
# ============================================================================

class PowerNormalization(nn.Module):
    """Normalize to average transmit power P."""
    def __init__(self, P=1.0):
        super().__init__()
        self.P = P
    
    def forward(self, x):
        power = torch.mean(x ** 2, dim=-1, keepdim=True)
        scale = torch.sqrt(self.P / (power + 1e-8))
        return x * scale