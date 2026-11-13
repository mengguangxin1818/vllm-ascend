# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker import model_runner_v1 as module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture
def runner():
    r = NPUModelRunner.__new__(NPUModelRunner)
    r.dp_size = 2
    r.dp_rank = 0
    r.dcp_size = 1
    r.uniform_decode_query_len = 4
    r.max_num_reqs = 32
    r.parallel_config = SimpleNamespace(pipeline_parallel_size=1, enable_dbo=False, data_parallel_rank=0)
    r.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, tensor_parallel_size=2),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    r.ascend_config = SimpleNamespace(
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=1,
            lmhead_tensor_parallel_size=8,
            embedding_tensor_parallel_size=1,
            mlp_tensor_parallel_size=1,
        )
    )
    r.input_batch = SimpleNamespace(
        num_reqs=1,
        num_computed_tokens_cpu=np.array([64, 0]),
        num_prompt_tokens=np.array([32, 999]),
        lora_id_to_lora_request={},
    )
    r.model_config = SimpleNamespace(is_hybrid=True, is_encoder_decoder=False)
    r._pad_for_sequence_parallelism = lambda n: n
    return r


@pytest.fixture(autouse=True)
def local_control(monkeypatch):
    monkeypatch.setattr(module, "get_dp_group", lambda: SimpleNamespace(cpu_group="dp-cpu"))
    monkeypatch.setattr(module, "should_skip_allreduce_across_dp_group", lambda *a, **k: False)
    monkeypatch.setattr(module, "select_moe_comm_method", lambda *a: MoECommType.FUSED_MC2)
    monkeypatch.setattr(module, "use_cann_megamoe", lambda *a: False)
    monkeypatch.setattr(module, "enable_sp", lambda *a: False)
    monkeypatch.setattr(module, "lmhead_tp_enable", lambda: True)
    monkeypatch.setattr(module, "record_function_or_nullcontext", lambda *a: nullcontext())


@pytest.mark.parametrize("size", [2, 8])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_async_collective_matches_synchronous_padding(runner, monkeypatch, size, mode):
    runner.dp_size = size
    events = []

    def complete(tensor):
        tensor[0, 1:] = 8
        tensor[1, :] = mode.value

    def reduce(tensor, *, group, async_op=False):
        assert group == "dp-cpu"
        assert tensor.shape == (2, size)
        assert tensor.dtype == torch.int32 and tensor.device.type == "cpu"
        if not async_op:
            complete(tensor)
            return None
        events.append("submit")

        def wait():
            events.append("wait")
            complete(tensor)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    baseline = runner._sync_metadata_across_dp(4, cudagraph_mode=mode)
    result = runner._sync_metadata_across_dp(
        4,
        cudagraph_mode=mode,
        prepare_inputs_during_sync=lambda: events.append("prepare"),
    )
    assert events == ["submit", "prepare", "wait"]
    assert result[0] == baseline[0] == 8
    assert result[2] == baseline[2] == mode
    torch.testing.assert_close(result[1], baseline[1])
    assert result[1].tolist() == [8] * size
    assert runner._get_lmhead_pad_size(result[1]) == 8


def test_preparation_error_drains_collective_and_next_step_has_no_stale_work(runner, monkeypatch):
    events = []
    work = SimpleNamespace(wait=lambda: events.append("wait"))
    reduce = Mock(return_value=work)
    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    error = RuntimeError("input preparation failed")

    def prepare():
        events.append("prepare")
        raise error

    with pytest.raises(RuntimeError) as caught:
        runner._sync_metadata_across_dp(4, prepare_inputs_during_sync=prepare)
    assert caught.value is error
    assert events == ["prepare", "wait"]
    reduce.assert_called_once()
    assert reduce.call_args.kwargs["async_op"] is True
    # Work is local to the call; the next step submits a fresh collective.
    runner._sync_metadata_across_dp(4)
    assert reduce.call_count == 2
    assert "async_op" not in reduce.call_args.kwargs


def test_collective_error_is_propagated_after_preparation(runner, monkeypatch):
    events = []
    work = SimpleNamespace(wait=Mock(side_effect=RuntimeError("collective failed")))
    monkeypatch.setattr(module.dist, "all_reduce", Mock(return_value=work))
    with pytest.raises(RuntimeError, match="collective failed"):
        runner._sync_metadata_across_dp(4, prepare_inputs_during_sync=lambda: events.append("prepare"))
    assert events == ["prepare"]
    work.wait.assert_called_once()


@pytest.mark.parametrize("fast_path", ["single_dp", "skip_sync"])
def test_no_collective_fast_paths_prepare_exactly_once(runner, monkeypatch, fast_path):
    if fast_path == "single_dp":
        runner.dp_size = 1
    else:
        monkeypatch.setattr(module, "should_skip_allreduce_across_dp_group", lambda *a: True)
    reduce = Mock()
    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    prepare = Mock()
    result = runner._sync_metadata_across_dp(4, prepare_inputs_during_sync=prepare)
    assert result[0] == 4
    prepare.assert_called_once_with()
    reduce.assert_not_called()


def test_draft_sync_remains_independent_and_synchronous(runner, monkeypatch):
    reduce = Mock()
    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    runner._sync_metadata_across_dp(4, is_draft_model=True)
    reduce.assert_called_once()
    assert "async_op" not in reduce.call_args.kwargs
    with pytest.raises(ValueError, match="main model"):
        runner._sync_metadata_across_dp(4, is_draft_model=True, prepare_inputs_during_sync=Mock())
    assert reduce.call_count == 1


@pytest.mark.parametrize("dp_size", [1, 2])
def test_final_graph_dispatch_waits_for_input_preparation(runner, monkeypatch, dp_size):
    events = []
    runner.dp_size = dp_size
    runner.vllm_config.parallel_config.data_parallel_size = dp_size
    runner.cudagraph_dispatcher = SimpleNamespace(
        dispatch=lambda **kw: (
            events.append(("dispatch", kw["num_tokens"])) or CUDAGraphMode.FULL,
            SimpleNamespace(num_tokens=kw["num_tokens"]),
        )
    )

    def reduce(tensor, *, group, async_op):
        assert async_op
        events.append("submit")

        def wait():
            events.append("wait")
            tensor[0, 1] = 8
            tensor[1, 1] = CUDAGraphMode.FULL.value

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    result = runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([4]),
        max_num_scheduled_tokens=4,
        use_cascade_attn=False,
        prepare_inputs_during_sync=lambda: events.append("prepare"),
    )
    if dp_size == 2:
        assert events == [("dispatch", 4), "submit", "prepare", "wait", ("dispatch", 8)]
        assert result[1].num_tokens == 8
    else:
        assert events == [("dispatch", 4), "prepare"]
        assert result[1].num_tokens == 4


def test_overlap_gate_supports_multi_group_graph_drafter_and_ignores_stale_rows(runner):
    # Eligibility is independent of proposer type and whether the MTP graph is enabled.
    runner.drafter = SimpleNamespace(use_cuda_graph=True, kv_cache_gid=0, draft_group_ids={0, 1})
    scheduler = SimpleNamespace(scheduled_encoder_inputs={})
    assert runner._can_overlap_dp_sync_with_input_prep(scheduler, np.array([4]))
    assert runner._can_overlap_dp_sync_with_input_prep(scheduler, np.array([1]))


@pytest.mark.parametrize(
    "unsupported",
    [
        "single_dp",
        "dcp",
        "pp",
        "dbo",
        "encoder",
        "prefill",
        "recompute_tail",
        "zero_state",
        "lora",
        "large_chunk",
        "zero_tokens",
        "no_lmhead_tp",
    ],
)
def test_unsupported_paths_keep_existing_order(runner, monkeypatch, unsupported):
    scheduler = SimpleNamespace(scheduled_encoder_inputs={})
    tokens = np.array([4])
    if unsupported == "single_dp":
        runner.dp_size = 1
    elif unsupported == "dcp":
        runner.dcp_size = 2
    elif unsupported == "pp":
        runner.parallel_config.pipeline_parallel_size = 2
    elif unsupported == "dbo":
        runner.parallel_config.enable_dbo = True
    elif unsupported == "encoder":
        scheduler.scheduled_encoder_inputs = {"r": [0]}
    elif unsupported in ("prefill", "recompute_tail", "zero_state"):
        runner.input_batch.num_computed_tokens_cpu[0] = {"prefill": 16, "recompute_tail": 31, "zero_state": 0}[
            unsupported
        ]
    elif unsupported == "lora":
        runner.input_batch.lora_id_to_lora_request = {1: object()}
    elif unsupported == "large_chunk":
        tokens[0] = 8
    elif unsupported == "zero_tokens":
        tokens[0] = 0
    elif unsupported == "no_lmhead_tp":
        monkeypatch.setattr(module, "lmhead_tp_enable", lambda: False)
    assert not runner._can_overlap_dp_sync_with_input_prep(scheduler, tokens)


@pytest.mark.parametrize("overlap", [False, True])
def test_execute_model_prepares_inside_existing_input_fence(runner, monkeypatch, overlap):
    from contextlib import contextmanager

    events = []
    fence_active = False
    result = SimpleNamespace()
    runner.uniform_decode_query_len = 1
    runner.input_batch.num_reqs = 2
    runner.input_batch.req_ids = ["a", "b"]
    runner.input_batch.num_computed_tokens_cpu = np.array([32, 32] if overlap else [0, 0])
    runner.input_batch.num_prompt_tokens = np.array([16, 16])
    runner.ascend_config.scheduler_config = SimpleNamespace(
        profiling_chunk_config=SimpleNamespace(enabled=False, need_timing=False),
    )
    runner.parallel_config.num_ubatches = 1
    runner.model_config.enforce_eager = False
    runner.model_config.hf_config = SimpleNamespace(model_type="test")
    runner.model_config.hf_text_config = SimpleNamespace(model_type="test")
    runner.execute_model_state = None
    runner.speculative_config = None
    runner.use_async_scheduling = False
    runner.use_compress = False
    runner.dynamic_eplb = False
    runner.cascade_attn_enabled = False
    runner._has_sinks = False
    runner.use_aux_hidden_state_outputs = False
    runner.broadcast_pp_output = False
    runner.is_pooling_model = True
    runner.model = object()
    runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False, mamba_cache_mode=None)

    @contextmanager
    def fence():
        nonlocal fence_active
        fence_active = True
        try:
            yield
        finally:
            fence_active = False

    runner.synchronize_input_prep = fence
    runner._update_states = lambda _: None
    runner._start_dump_data = Mock()
    runner._finalize_dump_data = Mock()

    def prepare(*args):
        assert fence_active
        events.append("prepare")
        return torch.tensor([0, 1]), None, 2

    def determine(**kwargs):
        events.append("local_dispatch")
        callback = kwargs.get("prepare_inputs_during_sync")
        if overlap:
            assert callback is not None
            events.append("submit")
            callback()
            events.append("wait")
        else:
            assert callback is None
        return CUDAGraphMode.NONE, SimpleNamespace(num_tokens=2, num_reqs=2), False, torch.tensor([2, 2]), None

    runner._prepare_inputs = prepare
    runner._determine_batch_execution_and_padding = determine
    runner._build_attention_metadata = lambda **kwargs: ({}, None)
    runner._sanitize_placeholder_input_ids_for_forward = Mock()
    runner._preprocess = lambda *args: (None, None, torch.arange(2), None, {}, None)
    runner._prepare_device_metadata_for_forward = lambda _: None
    runner.maybe_get_kv_connector_output = lambda *args, **kwargs: nullcontext()
    runner.kvpp = SimpleNamespace(scheduler=None, prepare_forward=lambda _: None, complete_forward=lambda: None)
    runner._model_forward = lambda *args, **kwargs: torch.ones(2, 1)
    runner._pool = lambda *args: result
    monkeypatch.setattr(module, "get_pp_group", lambda: SimpleNamespace(world_size=1, is_last_rank=True))
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(module, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(module, "maybe_create_ubatch_slices", lambda *args: (None, None))
    monkeypatch.setattr(module, "set_ascend_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(module, "update_cos_sin", Mock())
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"a": 1, "b": 1},
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
    )
    assert runner.execute_model(scheduler) is result
    assert not fence_active
    assert events == (["local_dispatch", "submit", "prepare", "wait"] if overlap else ["prepare", "local_dispatch"])
