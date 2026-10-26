"""Original HiVT methods, with only its local encoder augmented by pair GRU."""
import torch
from models.hivt import HiVT
from .encoder import GRULocalEncoder


class GRUOnlyHiVT(HiVT):
    def __init__(self, gru_enabled=True, relation_dim=16, relation_gain=0.1,
                 gru_only_version=1, **kwargs):
        if gru_only_version != 1:
            raise ValueError('Unsupported GRU-only checkpoint version')
        if gru_enabled and kwargs.get('parallel', False):
            raise ValueError('Pair-GRU requires parallel=False; original parallel code is not modified')
        # Build every original module first, in the original order.
        super().__init__(**kwargs)
        self.save_hyperparameters(dict(gru_enabled=bool(gru_enabled),
                                       relation_dim=relation_dim,
                                       relation_gain=relation_gain,
                                       gru_only_version=gru_only_version))
        if not gru_enabled:
            return
        original = self.local_encoder
        # New initialization must not change base weights or the global RNG stream.
        with torch.random.fork_rng(devices=[]):
            enhanced = GRULocalEncoder(
                historical_steps=self.historical_steps,
                node_dim=self.hparams.node_dim, edge_dim=self.hparams.edge_dim,
                embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout,
                num_temporal_layers=self.hparams.num_temporal_layers,
                local_radius=self.hparams.local_radius, parallel=False,
                relation_dim=relation_dim, relation_gain=relation_gain)
            result = enhanced.load_state_dict(original.state_dict(), strict=False)
        allowed = ('relation_memory.', 'aa_encoder.relation_logit.', 'aa_encoder.relation_value.')
        if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
            raise RuntimeError('Unexpected original model mismatch: ' + str(result))
        self.local_encoder = enhanced
    # forward / training_step / validation_step / configure_optimizers are inherited.
