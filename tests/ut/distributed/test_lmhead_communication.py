# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run standalone on CPU without the NPU-wide UT conftest:

python -m unittest discover -s tests/ut/distributed -p test_lmhead_communication.py -v
"""

import ast
import importlib.util
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# This module has no runtime vLLM dependencies. Loading it directly also permits
# standalone CPU checks without importing the plugin's NPU initialization.
_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/lmhead_communication.py"
_SPEC = importlib.util.spec_from_file_location("lmhead_communication_under_test", _SOURCE)
_COMMUNICATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMUNICATION)
configure_lmhead_alltoallv = _COMMUNICATION.configure_lmhead_alltoallv
gather_lmhead_hidden_states = _COMMUNICATION.gather_lmhead_hidden_states
scatter_lmhead_logits = _COMMUNICATION.scatter_lmhead_logits


def _load_logits_exchange(group):
    """Exercise the actual processor method without importing model/NPU loaders."""
    source = _SOURCE.parents[1] / "ops/vocab_parallel_embedding.py"
    tree = ast.parse(source.read_text())
    processor = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendLogitsProcessor"
    )
    method = next(
        node for node in processor.body if isinstance(node, ast.FunctionDef) and node.name == "_get_logits_lmheadtp"
    )
    namespace = {
        "torch": torch,
        "AscendParallelLMHead": torch.nn.Module,
        "get_lmhead_tp_group": lambda: group,
        "get_ascend_config": lambda: SimpleNamespace(enable_reduce_sample=False),
        "gather_lmhead_hidden_states": gather_lmhead_hidden_states,
        "scatter_lmhead_logits": scatter_lmhead_logits,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_get_logits_lmheadtp"]


class _TestGroup:
    """Small GroupCoordinator adapter using real Gloo/HCCL collectives."""

    def __init__(self, process_group):
        self.cpu_group = process_group
        self.device_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank_in_group = dist.get_rank(process_group)
        self.rank = self.rank_in_group

    def all_gather(self, tensor, dim=0):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor.contiguous(), group=self.device_group)
        return torch.cat(parts, dim=dim)

    def all_to_all(self, tensor, scatter_dim=0, gather_dim=-1, scatter_sizes=None, gather_sizes=None):
        if dist.get_backend(self.device_group) == "hccl":
            from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

            # Exercise the deployed list-based wrapper, including zero splits.
            return NPUCommunicator.all_to_all(self, tensor, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

        # Gloo lacks list-based all_to_all. Adapt the same row splits to its
        # all_to_all_single and retain source-rank vocabulary concatenation.
        assert scatter_dim == 0 and gather_dim == -1
        send_sizes = (
            scatter_sizes if scatter_sizes is not None else [tensor.shape[0] // self.world_size] * self.world_size
        )
        local_size = send_sizes[self.rank_in_group]
        received = tensor.new_empty((self.world_size * local_size, tensor.shape[-1]))
        if gather_sizes is not None:
            assert gather_sizes == [tensor.shape[-1]] * self.world_size
        dist.all_to_all_single(
            received,
            tensor.contiguous(),
            input_split_sizes=send_sizes,
            output_split_sizes=[local_size] * self.world_size,
            group=self.device_group,
        )
        return torch.cat(received.split([local_size] * self.world_size, dim=0), dim=-1)


def run_exchange_worker(rank, rendezvous, backend="gloo"):
    """Validate four-rank exchanges against unsharded GEMM, including idle ranks."""
    world_size = 4
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # noqa: F401

        torch.npu.set_device(rank)
        device = torch.device("npu", rank)
    dist.init_process_group(
        backend, init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    try:
        cpu_group = dist.new_group(backend="gloo") if backend == "hccl" else dist.group.WORLD
        group = _TestGroup(dist.group.WORLD)
        group.cpu_group = cpu_group
        compute_logits = _load_logits_exchange(group)
        # Change shapes on the same process group to expose stale metadata and
        # collective-order problems across MTP steps / real-dummy transitions.
        batches = ([1, 2, 0, 1], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 3, 0], [1, 1, 1, 7])
        hidden_size, shard_size = 5, 7
        weights = (
            torch.arange(world_size * shard_size * hidden_size, dtype=torch.float32, device=device).reshape(
                world_size * shard_size, hidden_size
            )
            / 100
        )
        for step, sizes in enumerate(batches):
            inputs = [
                (
                    torch.arange(size * hidden_size, dtype=torch.float32, device=device).reshape(size, hidden_size)
                    + source * 10
                    + step
                )
                for source, size in enumerate(sizes)
            ]
            gathered, actual_sizes = gather_lmhead_hidden_states(inputs[rank], group)
            assert actual_sizes == list(sizes)
            torch.testing.assert_close(gathered, torch.cat(inputs, dim=0))
            local_weights = weights[rank * shard_size : (rank + 1) * shard_size]
            logits = gathered @ local_weights.T
            actual = scatter_lmhead_logits(logits, actual_sizes, group)
            expected = inputs[rank] @ weights.T
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (sizes[rank], world_size * shard_size)
            # Preserve original vocabulary padding removal after redistribution.
            torch.testing.assert_close(actual[:, :-2], expected[:, :-2])
            if sizes[rank]:
                torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1))

            # Compare the graph-external processor branch with the original
            # padded path, including vocabulary trimming and bias.
            bias = torch.arange(world_size * shard_size, device=device, dtype=torch.float32)
            processor = SimpleNamespace(
                lmhead_alltoallv_enabled=True,
                org_vocab_size=world_size * shard_size - 2,
                head_dtype=None,
                _apply_head=lambda head, hidden, embedding_bias: torch.nn.functional.linear(
                    hidden, head.weight, embedding_bias
                ),
            )
            head = SimpleNamespace(weight=local_weights)
            local_bias = bias[rank * shard_size : (rank + 1) * shard_size]
            new_logits = compute_logits(processor, inputs[rank], head, local_bias)
            torch.testing.assert_close(new_logits, (expected + bias)[:, :-2])
            if sum(sizes):
                processor.lmhead_alltoallv_enabled = False
                padded = torch.nn.functional.pad(inputs[rank], (0, 0, 0, max(sizes) - sizes[rank]))
                original_logits = compute_logits(processor, padded, head, local_bias)
                torch.testing.assert_close(new_logits, original_logits[: sizes[rank]])
            else:
                # Even an empty result must honor the configured head dtype.
                processor.head_dtype = torch.float16
                empty_logits = compute_logits(processor, inputs[rank], head, local_bias)
                assert empty_logits.dtype == torch.float16

        # The deployed LMHead commonly uses BF16; also exercise its zero splits.
        sizes = [1, 0, 3, 2]
        hidden = torch.ones((sizes[rank], hidden_size), dtype=torch.bfloat16, device=device) * (rank + 1)
        gathered, actual_sizes = gather_lmhead_hidden_states(hidden, group)
        local_weights = weights[rank * shard_size : (rank + 1) * shard_size].to(torch.bfloat16)
        actual = scatter_lmhead_logits(gathered @ local_weights.T, actual_sizes, group)
        torch.testing.assert_close(actual, hidden @ weights.to(torch.bfloat16).T)
    finally:
        dist.destroy_process_group()


class TestLMHeadExchange(unittest.TestCase):
    def test_four_rank_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(run_exchange_worker, args=(f"file://{directory}/rendezvous",), nprocs=4, join=True)


class TestExistingPaddingPolicy(unittest.TestCase):
    """Restore fixed padding when the option is off; retain dynamic padding when on."""

    def _load_method(self, relative_path, class_name, method_name, skip, enabled):
        source = _SOURCE.parents[1] / relative_path
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
        namespace = {
            "torch": torch,
            "get_ascend_config": lambda: SimpleNamespace(enable_lmhead_alltoallv=enabled),
            "should_skip_allreduce_across_dp_group": lambda *args, **kwargs: skip,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        return namespace[method_name]

    def test_target_padding(self):
        runner = SimpleNamespace(max_num_reqs=32, uniform_decode_query_len=6, dcp_size=1, vllm_config=None)
        counts = torch.tensor([6, 12, 0, 6], dtype=torch.int32)
        for enabled in (False, True):
            for skip in (False, True):
                method = self._load_method(
                    "worker/model_runner_v1.py", "NPUModelRunner", "_get_lmhead_pad_size", skip, enabled
                )
                self.assertEqual(method(runner, counts), 12 if enabled and not skip else 192)
                self.assertEqual(method(runner, None), 192)
                runner.dcp_size = 2
                self.assertEqual(method(runner, counts), 192)
                runner.dcp_size = 1

    def test_mtp_padding(self):
        proposer = SimpleNamespace(
            method="mtp",
            dcp_size=1,
            vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=32)),
            runner=SimpleNamespace(uniform_decode_query_len=6),
        )
        for enabled in (False, True):
            for skip in (False, True):
                method = self._load_method(
                    "spec_decode/llm_base_proposer.py",
                    "AscendSpecDecodeBaseProposer",
                    "_get_lmhead_pad_size",
                    skip,
                    enabled,
                )
                self.assertEqual(method(proposer, 6), 6 if enabled and not skip else 192)
                self.assertEqual(method(proposer, 384), 192)
                proposer.dcp_size = 2
                self.assertEqual(method(proposer, 6), 192)
                proposer.dcp_size = 1


class TestLMHeadOption(unittest.TestCase):
    def test_config_defaults_and_conflicts(self):
        # Execute the real option-validation block without requiring the full
        # model configuration and hardware discovery used by AscendConfig.
        source = _SOURCE.parents[1] / "ascend_config.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")

        def assigns(node, attribute):
            return (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == attribute
            )

        start = next(index for index, node in enumerate(constructor.body) if assigns(node, "enable_reduce_sample"))
        end = next(index for index, node in enumerate(constructor.body) if assigns(node, "mix_placement"))
        code = compile(ast.Module(body=constructor.body[start:end], type_ignores=[]), str(source), "exec")
        for settings in ({}, {"enable_lmhead_alltoallv": True}, {"enable_reduce_sample": True}):
            config = SimpleNamespace()
            exec(code, {"self": config, "additional_config": settings})
            self.assertEqual(config.enable_lmhead_alltoallv, settings.get("enable_lmhead_alltoallv", False))
        for settings in (
            {"enable_lmhead_alltoallv": "true"},
            {"enable_lmhead_alltoallv": 1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                exec(code, {"self": SimpleNamespace(), "additional_config": settings})

        config = SimpleNamespace()
        logger = Mock()
        exec(
            code,
            {
                "self": config,
                "additional_config": {"enable_lmhead_alltoallv": True, "enable_reduce_sample": True},
                "logger": logger,
            },
        )
        logger.warning.assert_called_once_with(
            "enable_lmhead_alltoallv is inactive when enable_reduce_sample=true; using the existing reduce-sample path."
        )
        self.assertTrue(config.enable_reduce_sample)
        self.assertTrue(config.enable_lmhead_alltoallv)


class TestLMHeadConfiguration(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(enable_lmhead_alltoallv=True, enable_reduce_sample=False)
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
            ),
            lora_config=None,
        )
        self.processor_type = type("AscendLogitsProcessor", (torch.nn.Module,), {})
        self.model = torch.nn.Sequential(self.processor_type())
        ascend_config = ModuleType("vllm_ascend.ascend_config")
        ascend_config.get_ascend_config = lambda: self.config
        utils = ModuleType("vllm_ascend.utils")
        utils.lmhead_tp_enable = lambda: True
        utils.should_skip_allreduce_across_dp_group = lambda *args, **kwargs: True
        ops = ModuleType("vllm_ascend.ops.vocab_parallel_embedding")
        ops.AscendLogitsProcessor = self.processor_type
        self.modules = {
            "vllm_ascend.ascend_config": ascend_config,
            "vllm_ascend.utils": utils,
            "vllm_ascend.ops.vocab_parallel_embedding": ops,
        }
        if "vllm_ascend" not in sys.modules:
            package = ModuleType("vllm_ascend")
            package.__path__ = []
            self.modules["vllm_ascend"] = package
        self.module_patch = patch.dict("sys.modules", self.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def test_enable_target_and_draft(self):
        for is_draft_model in (False, True):
            with self.subTest(is_draft_model=is_draft_model):
                skip = self.modules["vllm_ascend.utils"]
                with patch.object(skip, "should_skip_allreduce_across_dp_group", return_value=True) as check:
                    self.assertTrue(
                        configure_lmhead_alltoallv(self.model, self.vllm_config, is_draft_model=is_draft_model)
                    )
                    check.assert_called_once_with(self.vllm_config, is_draft_model=is_draft_model)
                self.assertTrue(self.model[0].lmhead_alltoallv_enabled)

    def test_keep_existing_path(self):
        cases = [
            (self.config, "enable_lmhead_alltoallv", False),
            (self.config, "enable_reduce_sample", True),
            (self.vllm_config.parallel_config, "decode_context_parallel_size", 2),
            (self.vllm_config.parallel_config, "prefill_context_parallel_size", 2),
            (self.vllm_config.parallel_config, "pipeline_parallel_size", 2),
            (self.vllm_config, "lora_config", object()),
            (self.modules["vllm_ascend.utils"], "lmhead_tp_enable", lambda: False),
            (self.modules["vllm_ascend.utils"], "should_skip_allreduce_across_dp_group", lambda *args, **kwargs: False),
        ]
        for obj, name, value in cases:
            with self.subTest(name=name), patch.object(obj, name, value):
                self.assertFalse(configure_lmhead_alltoallv(self.model, self.vllm_config))
                self.assertFalse(getattr(self.model[0], "lmhead_alltoallv_enabled", False))

    def test_unsupported_processor(self):
        with self.assertRaisesRegex(ValueError, "requires AscendLogitsProcessor"):
            configure_lmhead_alltoallv(torch.nn.Identity(), self.vllm_config)

    def test_logits_as_input_is_rejected_before_enabling(self):
        self.model[0].logits_as_input = True
        with self.assertRaisesRegex(ValueError, "logits_as_input"):
            configure_lmhead_alltoallv(self.model, self.vllm_config)
        self.assertFalse(getattr(self.model[0], "lmhead_alltoallv_enabled", False))


if __name__ == "__main__":
    unittest.main()
