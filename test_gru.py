"""Synthetic CPU checks. Run from the ORIGINAL PLMT root; no dataset required."""
import ast
import inspect
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Batch
from models.hivt import HiVT
from models.local_encoder import AAEncoder
from utils import TemporalData, ImprovedDropEdge
from plmt_gru_only import GRUOnlyHiVT
from plmt_gru_only.encoder import GRUAAEncoder
from plmt_gru_only.memory import PairGRUMemory
from plmt_gru_only.compat import verify_original


def make_scene(n=3, offset=0.0, seq_id=1):
    """Small synthetic cache with displacement x, nonzero headings and a gap."""
    time = torch.arange(50).float() * 0.1
    positions = torch.zeros(n, 50, 2)
    for i in range(n):
        positions[i, :, 0] = offset + i * 5 + time * (3 + i) + 0.05 * time.square()
        positions[i, :, 1] = i * 1.5 + time * (0.1 + i * 0.15)
    edges = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j], dtype=torch.long)
    edges = edges.t().contiguous() if n > 1 else torch.empty(2, 0, dtype=torch.long)
    mask = torch.zeros(n, 50, dtype=torch.bool)
    if n > 2:
        mask[2, 3:5] = True
        positions[2, 3:5] = 0.0
    x = torch.zeros(n, 20, 2)
    x[:, 1:] = torch.where((mask[:, :19] | mask[:, 1:20]).unsqueeze(-1),
                          torch.zeros(n, 19, 2), positions[:, 1:20] - positions[:, :19])
    bos = torch.zeros(n, 20, dtype=torch.bool)
    bos[:, 0] = ~mask[:, 0]
    bos[:, 1:] = mask[:, :19] & ~mask[:, 1:20]
    heading = positions[:, 19] - positions[:, 18]
    angles = torch.atan2(heading[:, 1], heading[:, 0])
    velocity = torch.zeros(n, 50)
    velocity[:, :49] = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1) / 0.1
    velocity[:, 49] = velocity[:, 48]
    lane_vectors = torch.tensor([[3., 0.], [3., 0.2], [3., 0.4]])
    lane_start = torch.tensor([[offset, 1.], [offset + 3., 1.], [offset + 6., 1.2]])
    lane_edges = torch.tensor([(a, b) for a in range(3) for b in range(n)]).t().contiguous()
    return TemporalData(x=x, positions=positions, edge_index=edges,
                        y=positions[:, 20:] - positions[:, 19:20], num_nodes=n,
                        padding_mask=mask, bos_mask=bos, rotate_angles=angles,
                        lane_vectors=lane_vectors,
                        is_intersections=torch.zeros(3, dtype=torch.uint8),
                        turn_directions=torch.zeros(3, dtype=torch.uint8),
                        traffic_controls=torch.zeros(3, dtype=torch.uint8),
                        lane_actor_index=lane_edges,
                        lane_actor_vectors=lane_start[lane_edges[0]] - positions[lane_edges[1], 19],
                        seq_id=seq_id, agent_index=0, av_index=0, velocity=velocity)


def model_args(dropout=0.0):
    return dict(historical_steps=20, future_steps=30, num_modes=6, rotate=True,
                node_dim=2, edge_dim=2, embed_dim=16, num_heads=2, dropout=dropout,
                num_temporal_layers=1, num_global_layers=1, local_radius=50,
                parallel=False, lr=5e-4, weight_decay=1e-4, T_max=64)


def make_model(enabled=True, gain=0.1, dropout=0.0):
    torch.manual_seed(2022)
    return GRUOnlyHiVT(gru_enabled=enabled, relation_dim=16,
                       relation_gain=gain, **model_args(dropout))


def make_batch():
    return Batch.from_data_list([make_scene(3), make_scene(4, 100., 2)])


class StrictGRUOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_output_equal(self, first, second):
        for x, y in zip(first, second):
            self.assertTrue(torch.equal(x, y), float((x - y).abs().max()))

    def original(self, dropout=0.0):
        torch.manual_seed(2022)
        return HiVT(**model_args(dropout))

    def test_01_original_files_and_inherited_methods(self):
        verify_original(Path(inspect.getfile(HiVT)).resolve().parents[1])
        for method in ('forward', 'training_step', 'validation_step', 'configure_optimizers'):
            self.assertIs(getattr(GRUOnlyHiVT, method), getattr(HiVT, method))
        original_args = ast.parse(textwrap.dedent(inspect.getsource(AAEncoder.message))).body[0].args
        new_args = ast.parse(textwrap.dedent(inspect.getsource(GRUAAEncoder.message))).body[0].args
        self.assertEqual(ast.dump(original_args), ast.dump(new_args))
        original_source = textwrap.dedent(inspect.getsource(AAEncoder.message))
        expected = original_source.replace('alpha = softmax(alpha + Wvu, index, ptr, size_i)',
            "if self._relation_context is None:\n        raise RuntimeError('GRU attention used outside its local-encoder context')\n"
            '    relation = self._relation_context\n'
            '    alpha = softmax(alpha + Wvu + self.relation_logit(relation), index, ptr, size_i)')
        expected = expected.replace('return value * alpha.unsqueeze(-1)',
            'value = value + self.relation_value(relation).view(-1, self.num_heads, self.embed_dim // self.num_heads)\n'
            '    return value * alpha.unsqueeze(-1)')
        self.assertEqual(ast.dump(ast.parse(expected)),
                         ast.dump(ast.parse(textwrap.dedent(inspect.getsource(GRUAAEncoder.message)))))

    def test_02_off_exact_weights_rng_forward_loss_and_metrics(self):
        original = self.original()
        original_rng = torch.get_rng_state().clone()
        control = make_model(enabled=False)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()), set(control.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, control.state_dict()[key]), key)
        data = make_batch()
        original.eval(); control.eval()
        with torch.no_grad():
            self.assert_output_equal(original(data.clone()), control(data.clone()))
        for model in (original, control):
            model.log = lambda *a, **k: None
        first = original.training_step(data.clone(), 0)
        second = control.training_step(data.clone(), 0)
        self.assertTrue(torch.equal(first, second))
        with torch.no_grad():
            original.validation_step(data.clone(), 0)
            control.validation_step(data.clone(), 0)
        for name in ('minADE', 'minFDE', 'minMR'):
            self.assertTrue(torch.equal(getattr(original, name).compute(), getattr(control, name).compute()))

    def test_03_on_common_weights_rng_and_zero_injection_equivalence(self):
        original = self.original(dropout=0.1)
        original_rng = torch.get_rng_state().clone()
        enhanced = make_model(gain=0.0, dropout=0.1)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, enhanced.state_dict()[key]), key)
        data = make_batch()
        original.eval(); enhanced.eval()
        a, b = data.clone(), data.clone()
        with torch.no_grad():
            self.assert_output_equal(original(a), enhanced(b))
        for key in ('x', 'positions', 'velocity', 'padding_mask', 'bos_mask', 'y', 'rotate_mat'):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        original.train(); enhanced.train()
        original.log = enhanced.log = lambda *a, **k: None
        torch.manual_seed(500)
        first = original.training_step(data.clone(), 0)
        torch.manual_seed(500)
        second = enhanced.training_step(data.clone(), 0)
        self.assertTrue(torch.equal(first, second), (first.item(), second.item()))

    def test_04_filter_tuple_and_three_input_terms_unchanged_each_step(self):
        model = make_model().eval()
        encoder = model.local_encoder
        wrapper = encoder.imroved_drop_edge
        original_filter = wrapper.original
        original_aa_forward = encoder.aa_encoder.forward
        captured = []
        features = []
        memory_step = encoder.relation_memory.step

        def spy(edges, attrs, data, t):
            selected = original_filter(edges, attrs, data, t)
            independent = ImprovedDropEdge(1, 1, 1)(edges, attrs, data, t)
            self.assertEqual(len(selected), 4)
            for x, y in zip(selected, independent):
                self.assertTrue(torch.equal(x, y))
            self.assertEqual(selected[0].size(1), int(edges.size(1) * 0.8))
            row, col = edges
            distance = (data.positions[col, t] - data.positions[row, t]).norm(dim=-1)
            dv = (data.velocity[row, t] - data.velocity[col, t]).abs()
            theta = original_filter.vectorized_angle(data.positions[row, t + 1] - data.positions[row, t],
                data.positions[col, t + 1] - data.positions[col, t])
            captured.append((selected, torch.stack((distance / 50., dv / 10., theta.cos()), -1)))
            return selected

        class Spy:
            vectorized_angle = staticmethod(original_filter.vectorized_angle)
            def __call__(self, *args):
                return spy(*args)

        def step(edges, value):
            features.append(value.clone())
            memory_step(edges, value)
            # One stable state for every valid candidate, not only retained edges.
            self.assertEqual(encoder.relation_memory.gather(edges).shape[0], edges.size(1))

        def aa_forward(*args, **kwargs):
            selected, _ = captured[-1]
            for observed, expected in zip((kwargs['edge_index'], kwargs['edge_attr'],
                                           kwargs['theta'], kwargs['velocity']), selected):
                self.assertTrue(torch.equal(observed, expected))
            return original_aa_forward(*args, **kwargs)

        wrapper.original = Spy()
        encoder.relation_memory.step = step
        encoder.aa_encoder.forward = aa_forward
        try:
            with torch.no_grad():
                model(make_batch())
        finally:
            wrapper.original = original_filter
            encoder.relation_memory.step = memory_step
            encoder.aa_encoder.forward = original_aa_forward
        self.assertEqual(len(captured), 20)
        self.assertEqual(len(features), 20)
        for (_, expected), observed in zip(captured, features):
            self.assertTrue(torch.equal(expected, observed))

    def test_05_nonzero_gru_gradient_optimizer_coverage_and_update(self):
        model = make_model()
        model.log = lambda *a, **k: None
        optimizer = model.configure_optimizers()[0][0]
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        before = model.local_encoder.relation_memory.gru.weight_hh_l0.detach().clone()
        loss = model.training_step(make_batch(), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            if 'relation_memory.gru.' in name or 'aa_encoder.relation_' in name:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.local_encoder.relation_memory.gru.weight_hh_l0))

    def test_06_stable_pair_order_gap_reset_and_no_cross_forward_state(self):
        edges = torch.tensor([[2, 0, 1], [0, 1, 2]])
        memory = PairGRUMemory(4)
        memory.begin(edges, 3, torch.zeros(3, 20, 2))
        values = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
        memory.step(edges, values)
        first = memory.gather(edges).clone()
        self.assertTrue((first.abs().sum(-1) > 0).all())
        memory.begin(edges, 3, torch.zeros(3, 20, 2))
        permutation = torch.tensor([2, 0, 1])
        memory.step(edges[:, permutation], values[permutation])
        self.assertTrue(torch.equal(first, memory.gather(edges)))
        memory.step(edges[:, :1], values[:1])
        self.assertTrue(torch.equal(memory.gather(edges[:, 1:]), torch.zeros(2, 4)))
        memory.step(edges[:, 1:], values[1:])
        self.assertTrue(torch.allclose(memory.gather(edges[:, 1:]), first[1:], atol=1e-7, rtol=1e-6))
        memory.clear()
        model = make_model().eval()
        batch = make_batch()
        with torch.no_grad():
            a, b = model(batch.clone()), model(batch.clone())
        self.assert_output_equal(a, b)
        self.assertIsNone(model.local_encoder.relation_memory._state)
        self.assertIsNone(model.local_encoder.aa_encoder._relation_context)

    def test_07_single_actor_empty_edges_and_exception_cleanup(self):
        model = make_model().eval()
        with torch.no_grad():
            output = model(Batch.from_data_list([make_scene(1)]))
        for value in output:
            self.assertTrue(torch.isfinite(value).all())
        encoder = model.local_encoder
        original_filter = encoder.imroved_drop_edge.original
        def fail(*args):
            raise RuntimeError('injected test failure')
        encoder.imroved_drop_edge.original = fail
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected test failure'):
                model(make_batch())
            self.assertIsNone(encoder.relation_memory._state)
            self.assertIsNone(encoder.aa_encoder._relation_context)
        finally:
            encoder.imroved_drop_edge.original = original_filter

    def test_08_checkpoint_roundtrip(self):
        import pytorch_lightning as pl
        model = make_model().eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gru.ckpt'
            torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                        'pytorch-lightning_version': pl.__version__}, path)
            restored = GRUOnlyHiVT.load_from_checkpoint(str(path)).eval()
            self.assertTrue(restored.hparams.gru_enabled)
            with torch.no_grad():
                self.assert_output_equal(model(make_batch()), restored(make_batch()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
