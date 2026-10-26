"""Train original PLMT (off) or exactly that model plus pair GRU (on)."""
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT
from plmt_gru_only import GRUOnlyHiVT
from plmt_gru_only.compat import verify_original


def parse_switch(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise ValueError('Use true or false for --gru_enabled')


def main():
    parser = ArgumentParser(description=__doc__)
    # These original defaults and data-module arguments are unchanged.
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE',
                        choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--gru_enabled', type=parse_switch, default=True)
    parser.add_argument('--relation_dim', type=int, default=16)
    parser.add_argument('--relation_gain', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--log_root', default='gru_only_logs')
    parser.add_argument('--ckpt_path', default=None, help='Resume only this package, same on/off setting')
    parser.add_argument('--smoke', action='store_true', help='One training/validation batch; choose device with --gpus')
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.gru_enabled and args.parallel:
        parser.error('Use --parallel False is not supported by original bool parser; omit it (default False)')
    verify_original(Path(__file__).resolve().parent)
    if args.ckpt_path:
        import torch
        try:
            checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        previous = checkpoint.get('hyper_parameters', {})
        if previous.get('gru_only_version') != 1:
            parser.error('Do not resume the previous mixed-change package or an original baseline checkpoint')
        for key in ('gru_enabled', 'relation_dim', 'relation_gain', 'seed', 'embed_dim',
                    'historical_steps', 'future_steps', 'num_modes', 'num_heads',
                    'num_temporal_layers', 'num_global_layers', 'rotate', 'dropout',
                    'lr', 'weight_decay', 'T_max', 'train_batch_size', 'val_batch_size'):
            if previous.get(key) != getattr(args, key):
                parser.error('Resume setting changed: ' + key)
        del checkpoint
    pl.seed_everything(args.seed)
    model = GRUOnlyHiVT(**vars(args))
    data = ArgoverseV1DataModule.from_argparse_args(args)
    callbacks = [ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')]
    name = 'gru_on' if args.gru_enabled else 'baseline_off'
    logger = TensorBoardLogger(save_dir=args.log_root, name=name + '_seed' + str(args.seed))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger,
                                          fast_dev_run=args.smoke)
    print('GRU_ENABLED', args.gru_enabled, 'PARAMETERS', sum(p.numel() for p in model.parameters()))
    trainer.fit(model, data, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
