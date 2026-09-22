"""Evaluate the Swin + MAN density-map model."""

import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.crowd import Crowd
from models import swin_c_multibatch as models


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Swin + MAN')
    parser.add_argument('--model-name', choices=['swin_l_trans', 'swin_t_trans'],
                        default='swin_l_trans')
    parser.add_argument('--data-dir', default='SHA/clean')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--save-dir', required=True,
                        help='baseline or LoRA .pth/.tar checkpoint')
    parser.add_argument('--device', default='0')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--downsample-ratio', type=int, default=16)
    parser.add_argument('--lora-rank', type=int, default=0,
                        help='required when evaluating a LoRA .pth checkpoint')
    parser.add_argument('--lora-alpha', type=float, default=4.0)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(args.save_dir)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if checkpoint.get('model_name', args.model_name) != args.model_name:
            raise ValueError('Checkpoint model name does not match --model-name')
        state = checkpoint['model_state_dict']
        rank = checkpoint.get('lora_rank', args.lora_rank)
        alpha = checkpoint.get('lora_alpha', args.lora_alpha)
    else:
        state, rank, alpha = checkpoint, args.lora_rank, args.lora_alpha
    if not rank and any('.parametrizations.weight.' in key for key in state):
        raise ValueError('LoRA .pth checkpoint needs --lora-rank and --lora-alpha')

    model = getattr(models, args.model_name)(pretrained=False)
    if rank:
        model.enable_lora(rank, alpha)
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = Crowd(os.path.join(args.data_dir, args.split), args.crop_size,
                    args.downsample_ratio, is_gray=False, method='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    errors = []
    with torch.no_grad():
        for inputs, count, name in tqdm(loader, desc='Evaluate', dynamic_ncols=True):
            inputs = inputs.to(device)
            _, _, height, width = inputs.shape
            if height >= 3584 or width >= 3584:
                h_splits = math.ceil(height / 3584)
                w_splits = math.ceil(width / 3584)
                h_step = height // h_splits
                w_step = width // w_splits
                patches = [inputs[:, :, i*h_step:(i+1)*h_step if i+1<h_splits else height,
                                  j*w_step:(j+1)*w_step if j+1<w_splits else width]
                           for i in range(h_splits) for j in range(w_splits)]
            else:
                patches = [inputs]
            predicted = sum(model(patch)[0].sum().item() for patch in patches)
            errors.append(count.item() - predicted)
    print('MAE {:.3f} RMSE {:.3f}'.format(
        np.mean(np.abs(errors)), np.sqrt(np.mean(np.square(errors)))))


if __name__ == '__main__':
    main()
