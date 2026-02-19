import os
import sys
import datetime
import logging
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional

import dataloader
from metrics import PSNRLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))


def set_seed(seed=513):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser('Event-based Image Deblurring Training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_path', type=str, default='./deblur_log/', help='path to log')
    parser.add_argument('--train_path', type=str, default='./dataset/GOPRO/train/', help='train data path')
    parser.add_argument('--val_path', type=str, default='./dataset/GOPRO/test/', help='validation data path')
    parser.add_argument('--save_path', type=str, default='./checkpoint/gopro/', help='model save path')
    parser.add_argument('--pretrain_model_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
    parser.add_argument('--num_bins', default=12, type=int, help='number of event bins')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size for training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_res', default=20, type=int, help='number of residual blocks')
    return parser.parse_args()


def train(epoch, net, train_loader, optimizer, criterion, output_path):
    """Training function for one epoch"""
    net.train()
    total_loss = 0.0

    for batch_idx, (blur, sharp, event_frame) in enumerate(train_loader):
        blur, sharp, event_frame = blur.cuda(), sharp.cuda(), event_frame.cuda()
        
        optimizer.zero_grad()
        logits, spike_out, mask1 = net(blur, event_frame)
        
        # Multi-scale loss
        sharp_1 = F.interpolate(sharp, scale_factor=0.5)
        sharp_2 = F.interpolate(sharp_1, scale_factor=0.5)
        loss_max = criterion(logits[2] * 255., sharp * 255.)
        loss_mid = criterion(logits[1] * 255., sharp_1 * 255.)
        loss_min = criterion(logits[0] * 255., sharp_2 * 255.)
        loss = loss_max + loss_mid + loss_min
        
        loss.backward()
        optimizer.step()
        functional.reset_net(net)
        
        total_loss += loss.item()

        # Save training samples periodically
        if epoch % 10 == 0 and batch_idx == 0:
            with torch.no_grad():
                output_ = logits[2].permute(0, 2, 3, 1)
                output_ = torch.clamp(output_, min=0, max=1)
                output_np = output_.squeeze(0).cpu().detach().numpy()
                output_np = (output_np * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_path, f"train_output_epoch{epoch}.png"), output_np)

    return total_loss / len(train_loader)


def validate(epoch, net, val_loader, criterion, output_path):
    """Validation function"""
    net.eval()
    total_loss = 0.0
    total_psnr = []

    with torch.no_grad():
        for batch_idx, (blur, sharp, event_frame) in enumerate(val_loader):
            blur, sharp, event_frame = blur.cuda(), sharp.cuda(), event_frame.cuda()
            
            outputs, _, _ = net(blur, event_frame)
            functional.reset_net(net)

            # Save validation outputs
            if epoch % 10 == 0:
                output_ = outputs[2].permute(0, 2, 3, 1)
                output_ = torch.clamp(output_, min=0, max=1).cpu().detach().numpy()
                output_np = (output_[0] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_path, f"val_output_{batch_idx}_epoch{epoch}.png"), output_np)

            sharp_1 = F.interpolate(sharp, scale_factor=0.5)
            sharp_2 = F.interpolate(sharp_1, scale_factor=0.5)
            loss_max = criterion(outputs[2] * 255., sharp * 255.)
            loss_mid = criterion(outputs[1] * 255., sharp_1 * 255.)
            loss_min = criterion(outputs[0] * 255., sharp_2 * 255.)
            loss = loss_max + loss_mid + loss_min
            total_loss += loss.item()

            # Calculate PSNR
            output_cal = torch.clamp(outputs[2], min=0, max=1)
            mse = ((output_cal - sharp) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            total_psnr.append(psnr)

    avg_psnr = np.mean(total_psnr)
    return total_loss / len(val_loader), avg_psnr


def main(args):
    set_seed(513)
    
    # Setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    output_path = os.path.join(args.log_path, 'train_output')
    val_output_path = os.path.join(args.log_path, 'val_output')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.info(f'Arguments: {args}')

    # Data loading
    logger.info('Loading dataset...')
    train_dataset = dataloader.DataLoaderTrain_npz(rgb_dir=args.train_path, args=args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    val_dataset = dataloader.DataLoaderVal_npz(rgb_dir=args.val_path, args=args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    # Model loading
    from models.Model_gpu import BDHNet
    model = BDHNet(num_res=args.num_res, pretrained_path=args.pretrain_model_path)
    
    if not args.use_cpu:
        model = model.cuda()

    # Loss and optimizer
    criterion = PSNRLoss(loss_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80])

    # Tensorboard
    writer = SummaryWriter(args.log_path)

    # Training loop
    logger.info('Start training...')
    best_psnr = 0

    for epoch in range(args.epoch):
        scheduler.step(epoch)
        
        # Train
        train_loss = train(epoch, model, train_loader, optimizer, criterion, output_path)
        writer.add_scalar('Train/loss', train_loss, epoch)
        logger.info(f'Epoch {epoch+1}/{args.epoch}: Train Loss: {train_loss:.4f}')

        # Validate
        val_loss, avg_psnr = validate(epoch, model, val_loader, criterion, val_output_path)
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/psnr', avg_psnr, epoch)
        logger.info(f'Validation Loss: {val_loss:.4f}, PSNR: {avg_psnr:.4f}')

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_psnr.pth"))
            logger.info(f'Saved best model with PSNR: {best_psnr:.4f}')

        # Save checkpoint
        if epoch >= 60 and epoch <= 90:
            torch.save(model.state_dict(), os.path.join(args.save_path, f"{epoch}_checkpoint.pth"))
        
        torch.save(model.state_dict(), os.path.join(args.save_path, "checkpoint.pth"))

    writer.close()
    logger.info('Training completed!')


if __name__ == '__main__':
    args = parse_args()
    main(args)