import argparse
import numpy as np
import torch
import os
import sys
import logging
import importlib
import shutil
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

writer = SummaryWriter()

def parse_args():
    parser = argparse.ArgumentParser('TJSCC-PCT')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size [default: 8]')
    parser.add_argument('--model', default='tjscc_model', help='Model name')
    parser.add_argument('--revise', default=True, help='Coordinate re-estimation module')
    parser.add_argument('--snr', default=10, help='SNR of AWGN channel')
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs [default: 100]')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU instead of GPU')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate [default: 0.0001]')
    parser.add_argument('--gpu', type=str, default='7', help='GPU device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point number [default: 2048]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Use normal info [default: False]')
    parser.add_argument('--bottleneck_size', default=256, type=int)
    parser.add_argument('--recon_points', default=2048, type=int)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--dataset_path', type=str, default='/data/mohanad/shapenetcorev2')
    parser.add_argument('--multigpu', action='store_true', help='Enable multi-GPU training.')
    return parser.parse_args()

def test(args, model, loader):
    mean_loss = []
    mean_cd = []
    length = len(loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for j, data in enumerate(loader):
            if j % 100 == 0:
                print(j, '/', length)
            points = data[0].to(device)
            model.eval()
            _, cd, cbr = model(points, snr_db=args.snr)
            loss = cd
            mean_cd.append(cd.mean().item())
            mean_loss.append(loss.mean().item())
    return np.mean(mean_loss), np.mean(mean_cd)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories
    experiment_name = f"{args.bottleneck_size}_{args.recon_points}_snr{args.snr}"
    experiment_dir = Path('./log/') / args.model / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir / 'checkpoints'
    log_dir = experiment_dir / 'logs'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir / f'{args.model}.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS...')
    log_string(str(args))

    # Dataset loading
    log_string('Loading dataset...')
    from dataset import Dataset
    TRAIN_DATASET = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='train')
    VAL_DATASET = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='val')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model loading
    MODEL = __import__(args.model)
    model = MODEL.TJSCC_PCT(normal_channel=False, bottleneck_size=args.bottleneck_size).to(device)
    print(model)

    # Load pretrained or start fresh
    start_epoch = 0
    best_loss_test = float('inf')
    try:
        checkpoint = torch.load(checkpoints_dir / 'best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Using pre-trained model')
    except Exception as e:
        log_string('No existing model, starting from scratch')

    try:
        if args.pretrained:
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            log_string('Fine-tuning from pretrained model')
    except Exception as e:
        log_string('No pretrained model provided')

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    global_step = 0

    # Training loop
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        mean_loss = []
        mean_cd_loss = []
        log_string(f'Epoch {epoch + 1} ({epoch + 1}/{args.epoch}):')
        length = len(trainDataLoader)

        for batch_id, data in enumerate(trainDataLoader, 0):
            if batch_id % 100 == 0:
                print(f'{batch_id} / {length}')

            points = data[0].to(device)
            optimizer.zero_grad()
            model.train()

            _, cd, cbr = model(points, snr_db=args.snr)


            loss = cd  # Avoid multiplying by large numbers like 1000

            if args.multigpu:
                loss = loss.mean()
                cd = cd.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            mean_loss.append(loss.item())
            mean_cd_loss.append(cd.item())
            global_step += 1

        ml = np.mean(mean_loss)
        mcd = np.mean(mean_cd_loss)
        log_string(f'mean loss: {ml}')
        log_string(f'mean chamfer distance: {mcd}')
        writer.add_scalar('Train/loss', ml, epoch)
        writer.add_scalar('Train/chamfer_dist', mcd, epoch)

        if (epoch % 2 == 0) or (epoch > 50):
            log_string('Start validation...')
            model.eval()
            mean_loss_test, mean_cd_test = test(args, model, valDataLoader)
            log_string(f'val loss: {mean_loss_test}')
            log_string(f'val cd: {mean_cd_test}')
            writer.add_scalar('Validate/loss', mean_loss_test, epoch)
            writer.add_scalar('Validate/chamfer_dist', mean_cd_test, epoch)

            if mean_loss_test < best_loss_test:
                best_loss_test = mean_loss_test
                savepath = str(checkpoints_dir / 'best_model.pth')
                log_string(f'Saving model at {savepath}')
                state = {
                    'epoch': epoch,
                    'loss': mean_loss_test,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        scheduler.step()  # Moved after all batches

    logger.info('End of training.')

if __name__ == '__main__':
    args = parse_args()
    main(args)