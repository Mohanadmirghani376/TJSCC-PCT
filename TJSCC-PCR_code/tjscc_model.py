"""
TJSCC-PCT: Main Model and Testing
Imports components from tjscc_modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from tjscc_modules import (
    InputEmbedding,
    TransformerBlock,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
    WirelessChannel,
    PowerNormalization
)

# ============================================================================
# MAIN TJSCC-PCT MODEL
# ============================================================================

class TJSCC_PCT(nn.Module):
    """
    Transformer-Based Joint Source-Channel Coding for 3D Point Cloud Transmission
    Aligned with paper specifications (Section III & IV)
    """
    def __init__(self, normal_channel=False, bottleneck_size=256, channel_mode='awgn'):
        super().__init__()
        
        self.bottleneck_size = bottleneck_size
        self.channel_mode = channel_mode
        input_dim = 6 if normal_channel else 3
        
        # ==================== ENCODER (4 stages) ====================
        self.input_embedding = InputEmbedding(in_channel=input_dim, out_channel=128)
        
        # Stage 1: 2048→1024 points
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 128, [64, 64, 128])
        self.attn1 = TransformerBlock(128, 128, k=32, num_heads=4)
        
        # Stage 2: 1024→512 points
        self.sa2 = PointNetSetAbstraction(512, 0.2, 32, 128, [128, 128, 256])
        self.attn2 = TransformerBlock(256, 256, k=32, num_heads=4)
        
        # Stage 3: 512→256 points
        self.sa3 = PointNetSetAbstraction(256, 0.4, 32, 256, [256, 256, 512])
        self.attn3 = TransformerBlock(512, 512, k=32, num_heads=4)
        
        # Stage 4: 256→128 points
        self.sa4 = PointNetSetAbstraction(128, 0.6, 32, 512, [512, 512, bottleneck_size])
        self.attn4 = TransformerBlock(bottleneck_size, bottleneck_size, k=32, num_heads=4)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ==================== CHANNEL LAYER ====================
        self.fc_encoder = nn.Linear(bottleneck_size, bottleneck_size)
        self.pwr_norm = PowerNormalization(P=1.0)
        self.channel = WirelessChannel(mode=channel_mode)
        self.fc_decoder = nn.Linear(bottleneck_size, bottleneck_size)
        
        # ==================== DECODER (4 stages) ====================
        self.fp4 = PointNetFeaturePropagation(bottleneck_size + 512, [512, 512])
        self.dec_attn4 = TransformerBlock(512, 512, k=32, num_heads=4)
        
        self.fp3 = PointNetFeaturePropagation(512 + 256, [512, 256])
        self.dec_attn3 = TransformerBlock(256, 256, k=32, num_heads=4)
        
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.dec_attn2 = TransformerBlock(128, 128, k=32, num_heads=4)
        
        self.fp1 = PointNetFeaturePropagation(128 + 128, [128, 64])
        self.dec_attn1 = TransformerBlock(64, 64, k=32, num_heads=4)
        
        self.xyz_proj = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
    
    def forward(self, xyz, snr_db=10.0):
        B, N, _ = xyz.shape
        
        # ==================== ENCODING ====================
        l0_xyz, l0_feat = self.input_embedding(xyz, None)
        
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l1_feat, _ = self.attn1(l1_xyz, l1_feat)
        
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l2_feat, _ = self.attn2(l2_xyz, l2_feat)
        
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        l3_feat, _ = self.attn3(l3_xyz, l3_feat)
        
        l4_xyz, l4_feat = self.sa4(l3_xyz, l3_feat)
        l4_feat, _ = self.attn4(l4_xyz, l4_feat)
        
        latent = self.global_pool(l4_feat.permute(0, 2, 1)).squeeze(-1)
        
        # ==================== CHANNEL ====================
        t = self.fc_encoder(latent)
        s = self.pwr_norm(t)
        CBR = self.bottleneck_size / (3 * N)
        y = self.channel(s, snr_db)
        z_hat = self.fc_decoder(y).unsqueeze(1)
        
        # ==================== DECODING ====================
        fp4 = self.fp4(l3_xyz, l4_xyz, l3_feat, z_hat)
        fp4, _ = self.dec_attn4(l3_xyz, fp4)
        
        fp3 = self.fp3(l2_xyz, l3_xyz, l2_feat, fp4)
        fp3, _ = self.dec_attn3(l2_xyz, fp3)
        
        fp2 = self.fp2(l1_xyz, l2_xyz, l1_feat, fp3)
        fp2, _ = self.dec_attn2(l1_xyz, fp2)
        
        fp1 = self.fp1(l0_xyz, l1_xyz, l0_feat, fp2)
        fp1, _ = self.dec_attn1(l0_xyz, fp1)
        
        recon = self.xyz_proj(fp1.permute(0, 2, 1)).permute(0, 2, 1)
        
        cd_loss = chamfer_distance(xyz, recon)[0]
        
        return recon, cd_loss, CBR
    
    def get_num_params(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = TJSCC_PCT(bottleneck_size=256, channel_mode='awgn').to(device)
    

    
    # Test input
    batch_size = 2
    N = 2048
    xyz = torch.randn(batch_size, N, 3).to(device)
    xyz = xyz / (torch.max(torch.norm(xyz, dim=-1, keepdim=True)) + 1e-8)
    
    # Test training mode
    print("\nTesting training mode...")
    model.train()
    recon, loss, cbr = model(xyz, snr_db=10.0)
    print(f"Training Mode: Loss={loss.item():.4f}")
    loss.backward()
    print("✓ Backpropagation successful")
    
    # Test Rayleigh fading
    print("\nTesting Rayleigh fading...")
    model_fading = TJSCC_PCT(bottleneck_size=256, channel_mode='rayleigh').to(device)
    model_fading.eval()
    with torch.no_grad():
        recon, loss, cbr = model_fading(xyz, snr_db=10)
    print(f"Rayleigh Fading: CD Loss={loss.item():.4f}")
    
    print("\n✓ All tests passed!")