"""Original dataset and validation method; sequential path matches training."""
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from datasets import ArgoverseV1Dataset
from plmt_gru_only import GRUOnlyHiVT
from plmt_gru_only.compat import verify_original


def main():
    pl.seed_everything(2022)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', required=True)
    args = parser.parse_args()
    verify_original(Path(__file__).resolve().parent)
    model = GRUOnlyHiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    if model.hparams.get('gru_only_version') != 1:
        raise ValueError('Expected this GRU-only package checkpoint')
    data = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory,
                        persistent_workers=args.persistent_workers)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.validate(model, loader)


if __name__ == '__main__':
    main()
