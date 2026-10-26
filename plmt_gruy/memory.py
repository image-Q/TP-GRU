"""Transient directed-pair memory; no changes to original motion features."""
import torch
from torch import nn


class PairGRUMemory(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError('relation_dim must be positive')
        self.hidden_dim = hidden_dim
        self.embed = nn.Sequential(nn.Linear(3, hidden_dim), nn.Tanh())
        # Original HiVT optimizer already classifies nn.GRU. Do not change it.
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.clear()

    def clear(self):
        self._state = None
        self._sorted_keys = None
        self._slots = None
        self._num_nodes = None

    def begin(self, edge_index, num_nodes, reference):
        self._num_nodes = num_nodes
        keys = edge_index[0] * num_nodes + edge_index[1]
        self._sorted_keys, self._slots = torch.sort(keys)
        if keys.numel() > 1 and (self._sorted_keys[1:] == self._sorted_keys[:-1]).any():
            raise ValueError('GRU requires one slot per directed pair')
        self._state = reference.new_zeros(keys.numel(), self.hidden_dim)

    def slots(self, edge_index):
        if self._state is None:
            raise RuntimeError('Pair memory used outside forward')
        keys = edge_index[0] * self._num_nodes + edge_index[1]
        if not keys.numel():
            return keys
        index = torch.searchsorted(self._sorted_keys, keys)
        if (index >= self._sorted_keys.numel()).any():
            raise ValueError('Unknown directed pair')
        if (self._sorted_keys[index] != keys).any():
            raise ValueError('Unknown directed pair')
        return self._slots[index]

    def step(self, edge_index, features):
        slots = self.slots(edge_index)
        # Absent (invalid-position) pairs reset; valid unselected pairs still update.
        next_state = torch.zeros_like(self._state)
        if slots.numel():
            embedded = self.embed(features)
            _, hidden = self.gru(embedded.unsqueeze(0), self._state[slots].unsqueeze(0))
            next_state = next_state.index_copy(0, slots, hidden.squeeze(0))
        self._state = next_state

    def gather(self, edge_index):
        return self._state[self.slots(edge_index)]
