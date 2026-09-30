#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import sitecustomize
import torch

import models_mae
from main_pretrain import model_state_sha256


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Create a model-only MAE initialization checkpoint.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='mae_vit_base_patch16')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true')
    return parser


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model_sha256 = model_state_sha256(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(
            'Refusing to overwrite existing initialization: {}'.format(args.output))

    torch.save({
        'model': model.state_dict(),
        'model_name': args.model,
        'norm_pix_loss': args.norm_pix_loss,
        'seed': args.seed,
        'model_sha256': model_sha256,
    }, args.output)
    print('Wrote {}'.format(args.output))
    print('Model SHA256: {}'.format(model_sha256))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
