# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified 2026-10-16: add pair GRU context; original AA terms preserved.
"""Original local-encoder forward and AA aggregation, plus pair GRU only.

AA message is copied from the reviewed Apache-2.0 HiVT-derived source, with
only the relation additions shown in the package's message_diff.patch.
Original copyright: 2022 Zikang Zhou. Original project files are not replaced.
"""
from typing import Optional
import torch
from torch import nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from models.local_encoder import LocalEncoder, AAEncoder
from .memory import PairGRUMemory


class GRUAAEncoder(AAEncoder):
    def __init__(self, *args, relation_dim=16, relation_gain=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_logit = nn.Linear(relation_dim, self.num_heads, bias=False)
        self.relation_value = nn.Linear(relation_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.relation_logit.weight, gain=relation_gain)
        nn.init.xavier_uniform_(self.relation_value.weight, gain=relation_gain)
        self._relation_context = None

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int],
                theta,
                velocity) -> torch.Tensor:
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed([torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                                        torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        Wvu = self.beta * torch.cos(theta) - self.gamma * velocity
        Wvu = Wvu.reshape(-1, 1)
        if self._relation_context is None:
            raise RuntimeError('GRU attention used outside its local-encoder context')
        relation = self._relation_context
        alpha = softmax(alpha + Wvu + self.relation_logit(relation), index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        value = value + self.relation_value(relation).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        return value * alpha.unsqueeze(-1)


class _OriginalDropWithGRU:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, edge_index, edge_attr, data, t):
        # Call the original batch-global filter exactly once. Return its exact
        # tuple, without recomputing scores, changing membership or reordering.
        selected = self.original(edge_index, edge_attr, data, t)
        row, col = edge_index
        positions, velocity = data['positions'], data['velocity']
        pi1, pi2 = positions[row, t], positions[row, t + 1]
        pj1, pj2 = positions[col, t], positions[col, t + 1]
        distance = torch.norm(pj1 - pi1, p=2, dim=-1)
        delta_speed = torch.abs(velocity[row, t] - velocity[col, t])
        theta = self.original.vectorized_angle(pi2 - pi1, pj2 - pj1)
        # Only new GRU inputs are scaled. Original theta, dv, score and data
        # remain untouched. These are exactly the three original physical terms.
        features = torch.stack((distance / 50.0, delta_speed / 10.0, torch.cos(theta)), dim=-1)
        self.owner.relation_memory.step(edge_index, features)
        self.owner.aa_encoder._relation_context = self.owner.relation_memory.gather(selected[0])
        return selected


class GRULocalEncoder(LocalEncoder):
    def __init__(self, historical_steps, node_dim, edge_dim, embed_dim,
                 num_heads=8, dropout=0.1, num_temporal_layers=4,
                 local_radius=50, parallel=False, relation_dim=16,
                 relation_gain=0.1):
        if parallel:
            raise ValueError('Pair-GRU requires the original sequential local path')
        super().__init__(historical_steps, node_dim, edge_dim, embed_dim,
                         num_heads, dropout, num_temporal_layers, local_radius, parallel)
        self.aa_encoder = GRUAAEncoder(
            historical_steps=historical_steps, node_dim=node_dim, edge_dim=edge_dim,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, parallel=parallel,
            relation_dim=relation_dim, relation_gain=relation_gain)
        self.relation_memory = PairGRUMemory(relation_dim)
        self.imroved_drop_edge = _OriginalDropWithGRU(self.imroved_drop_edge, self)

    def forward(self, data):
        try:
            self.relation_memory.begin(data.edge_index, data.num_nodes, data.x)
            # Original subgraph, physical selection, temporal encoder and map
            # interaction run unchanged. Only the wrapper/message add GRU.
            return super().forward(data)
        finally:
            self.aa_encoder._relation_context = None
            self.relation_memory.clear()
