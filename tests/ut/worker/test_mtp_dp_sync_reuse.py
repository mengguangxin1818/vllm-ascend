# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.spec_decode.dp_sync_reuse import MTPDPSyncReuse, MTPDPSyncRole
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import (
    AscendMultiKVCacheGroupMTPProposer,
)
from vllm_ascend.worker import model_runner_v1 as module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture
def runner(monkeypatch):
    r = NPUModelRunner.__new__(NPUModelRunner)
    r.dp_size = 2
    r.dp_rank = 0
    r.dcp_size = 1
    r.num_spec_tokens = 3
    r.effective_drafter_max_model_len = 1024
    r.drafter = SimpleNamespace(
        supports_mtp_dp_sync_reuse=True,
        needs_extra_input_slots=False,
        parallel_drafting=False,
    )
    r.speculative_config = SimpleNamespace(disable_padded_drafter_batch=False)
    r.input_batch = SimpleNamespace(num_reqs=1, num_computed_tokens_cpu=np.array([64]))
    r.ascend_config = SimpleNamespace(
        finegrained_tp_config=SimpleNamespace(
            oproj_tensor_parallel_size=1,
            lmhead_tensor_parallel_size=8,
            embedding_tensor_parallel_size=1,
            mlp_tensor_parallel_size=1,
        )
    )
    r.vllm_config = object()
    monkeypatch.setattr(module.ascend_envs, "VLLM_ASCEND_MERGE_MTP_DP_SYNC", True)
    monkeypatch.setattr(module, "get_dp_group", lambda: SimpleNamespace(cpu_group="dp"))
    monkeypatch.setattr(module, "should_skip_allreduce_across_dp_group", lambda *a: False)
    monkeypatch.setattr(module, "select_moe_comm_method", lambda *a: MoECommType.FUSED_MC2)
    monkeypatch.setattr(module, "use_cann_megamoe", lambda *a: False)
    return r


@pytest.mark.parametrize("roles", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_reuse_first_draft_once_with_active_and_dummy_ranks(roles):
    reuse = MTPDPSyncReuse()
    reuse.main_completed(
        full_graph=True,
        predicted_tokens=[4, 0 if roles[1] == 2 else 4],
        roles=roles,
        main_padded_tokens=[4, 4],
        reuse_flags=[1, 1],
    )
    assert reuse.consume_prediction(rank=0, local_tokens=4, mode=0) == (4, (4, 4))
    assert reuse.consume_prediction(rank=0, local_tokens=4, mode=0) is None


@pytest.mark.parametrize(
    "fault",
    ["unsupported_rank", "disabled_rank", "eager", "large", "unequal", "padding"],
)
def test_group_falls_back_outside_validated_conditions(fault):
    reuse = MTPDPSyncReuse()
    args = dict(
        full_graph=True,
        predicted_tokens=[4, 4],
        roles=[1, 1],
        main_padded_tokens=[4, 4],
        reuse_flags=[1, 1],
    )
    if fault == "unsupported_rank":
        args["roles"][1] = 0
    elif fault == "disabled_rank":
        args["reuse_flags"][1] = 0
    elif fault == "eager":
        args["full_graph"] = False
    elif fault == "large":
        args["predicted_tokens"] = [8, 8]
        args["main_padded_tokens"] = [8, 8]
    elif fault == "unequal":
        args["predicted_tokens"] = [4, 8]
    else:
        args["main_padded_tokens"] = [8, 8]
    reuse.main_completed(**args)
    assert reuse.consume_prediction(rank=0, local_tokens=4, mode=0) is None


@pytest.mark.parametrize("reset", ["begin_step", "clear_prediction", "unsupported_main"])
def test_prediction_does_not_leak_into_later_rounds(reset):
    reuse = MTPDPSyncReuse()
    reuse.main_completed(
        full_graph=True,
        predicted_tokens=[4, 4],
        roles=[1, 1],
        main_padded_tokens=[4, 4],
        reuse_flags=[1, 1],
    )
    if reset == "unsupported_main":
        reuse.main_completed(
            full_graph=True,
            predicted_tokens=[4, 0],
            roles=[1, 0],
            main_padded_tokens=[4, 4],
            reuse_flags=[1, 1],
        )
    else:
        getattr(reuse, reset)()
    assert reuse.consume_prediction(rank=0, local_tokens=4, mode=0) is None


@pytest.mark.parametrize("local_tokens,mode", [(8, 0), (4, 2)])
def test_contract_error_fails_before_unilateral_collective(local_tokens, mode):
    reuse = MTPDPSyncReuse()
    reuse.main_completed(
        full_graph=True,
        predicted_tokens=[4, 4],
        roles=[1, 1],
        main_padded_tokens=[4, 4],
        reuse_flags=[1, 1],
    )
    with pytest.raises(RuntimeError, match="contract violated"):
        reuse.consume_prediction(rank=0, local_tokens=local_tokens, mode=mode)
    assert reuse.pending is None


@pytest.mark.parametrize("peer_allows", [False, True])
@pytest.mark.parametrize("local_allows", [False, True])
def test_collective_skipped_only_with_whole_group_consent(runner, monkeypatch, peer_allows, local_allows):
    monkeypatch.setattr(module.ascend_envs, "VLLM_ASCEND_MERGE_MTP_DP_SYNC", local_allows)
    reuse = runner._get_mtp_dp_sync_reuse()
    reuse.begin_step()
    reuse.predict_active(4)
    calls = []

    def reduce(tensor, **kwargs):
        calls.append(tuple(tensor.shape))
        tensor[0, 1] = 4
        if tensor.shape[0] == 5:
            tensor[1, 1] = CUDAGraphMode.FULL.value
            tensor[2, 1] = 4
            tensor[3, 1] = MTPDPSyncRole.ACTIVE
            tensor[4, 1] = int(peer_allows)

    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    main = runner._sync_metadata_across_dp(4, cudagraph_mode=CUDAGraphMode.FULL)
    draft = runner._sync_metadata_across_dp(4, is_draft_model=True)
    assert main[0] == draft[0] == 4
    assert main[1].tolist() == draft[1].tolist() == [4, 4]
    assert main[2] == CUDAGraphMode.FULL and draft[2] == CUDAGraphMode.NONE
    assert calls == ([(5, 2)] if local_allows and peer_allows else [(5, 2), (2, 2)])
    runner._sync_metadata_across_dp(4, is_draft_model=True)
    assert calls[-1] == (2, 2)  # no prediction is consumed twice


def test_unsupported_proposer_preserves_two_row_packet(runner, monkeypatch):
    runner.drafter.supports_mtp_dp_sync_reuse = False
    calls = []
    monkeypatch.setattr(
        module.dist,
        "all_reduce",
        lambda tensor, **kwargs: calls.append(tuple(tensor.shape)),
    )
    runner._sync_metadata_across_dp(4)
    runner._sync_metadata_across_dp(4, is_draft_model=True)
    assert calls == [(2, 2), (2, 2)]


@pytest.mark.parametrize("case", ["prefill", "unpadded", "dynamic_k", "extra", "near_limit", "disabled"])
def test_unsupported_local_batches_veto_group_reuse(runner, monkeypatch, case):
    scheduler = SimpleNamespace(num_spec_tokens_to_schedule=3)
    reuse = runner._get_mtp_dp_sync_reuse()
    reuse.predict_active(4)
    if case == "unpadded":
        runner.speculative_config.disable_padded_drafter_batch = True
    elif case == "dynamic_k":
        scheduler.num_spec_tokens_to_schedule = 0
    elif case == "extra":
        runner.drafter.needs_extra_input_slots = True
    elif case == "near_limit":
        runner.input_batch.num_computed_tokens_cpu[0] = 1020
    elif case == "disabled":
        monkeypatch.setattr(module.ascend_envs, "VLLM_ASCEND_MERGE_MTP_DP_SYNC", False)
    runner._prepare_mtp_dp_sync_reuse(scheduler, np.array([4]), case != "prefill")
    assert reuse.local_fields() == (0, MTPDPSyncRole.UNSUPPORTED)


@pytest.mark.parametrize("graph", [False, True])
def test_graph_prediction_uses_existing_dispatcher_without_kv_mutation(graph):
    dispatch = Mock(return_value=(CUDAGraphMode.FULL, SimpleNamespace(num_tokens=8)))
    drafter = SimpleNamespace(
        use_cuda_graph=graph,
        needs_extra_input_slots=False,
        parallel_drafting=False,
        runner=SimpleNamespace(cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch)),
    )
    value = AscendMultiKVCacheGroupMTPProposer.predict_first_dp_sync_tokens(drafter, 6, True, False)
    assert value == (8 if graph else 6)
    assert dispatch.call_count == int(graph)
