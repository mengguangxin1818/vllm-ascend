# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator


def configure_lmhead_alltoallv(
    model: torch.nn.Module,
    vllm_config: "VllmConfig",
    *,
    is_draft_model: bool = False,
) -> bool:
    """Opt a model's logits processors into variable-size, graph-external exchange.

    Call only for the target's post-forward logits or an eager MTP model.
    Model-local state avoids changing the shared LMHead weights or other models.
    """
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import lmhead_tp_enable, should_skip_allreduce_across_dp_group

    config = get_ascend_config()
    parallel = vllm_config.parallel_config
    if not (
        config.enable_lmhead_alltoallv
        and not config.enable_reduce_sample
        and lmhead_tp_enable()
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and vllm_config.lora_config is None
        and should_skip_allreduce_across_dp_group(vllm_config, is_draft_model=is_draft_model)
    ):
        return False

    # Local import avoids a cycle with AscendLogitsProcessor.
    from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor

    processors = [module for module in model.modules() if isinstance(module, AscendLogitsProcessor)]
    if not processors:
        raise ValueError("enable_lmhead_alltoallv requires AscendLogitsProcessor")
    if any(getattr(processor, "logits_as_input", False) for processor in processors):
        raise ValueError("enable_lmhead_alltoallv does not support logits_as_input")
    for processor in processors:
        processor.lmhead_alltoallv_enabled = True
    return True


def gather_lmhead_hidden_states(
    hidden_states: torch.Tensor, group: "GroupCoordinator"
) -> tuple[torch.Tensor, list[int]]:
    """Collect valid rows, using AllToAllV as AllGatherV for unequal lengths."""
    # Shapes are host metadata; this does not read a tensor back from the NPU.
    local_size = torch.tensor([hidden_states.shape[0]], dtype=torch.int32, device="cpu")
    lengths = [torch.empty_like(local_size) for _ in range(group.world_size)]
    dist.all_gather(lengths, local_size, group=group.cpu_group)
    sizes = [int(length.item()) for length in lengths]
    if sum(sizes) == 0:
        return hidden_states, sizes
    if all(size == sizes[0] for size in sizes):
        return group.all_gather(hidden_states, dim=0), sizes

    # Each vocabulary shard needs every rank's rows. all_to_all_single expects
    # one contiguous send segment per destination, including our own rank.
    send = hidden_states.repeat(group.world_size, 1)
    gathered = hidden_states.new_empty((sum(sizes), hidden_states.shape[1]))
    dist.all_to_all_single(
        gathered,
        send,
        output_split_sizes=sizes,
        input_split_sizes=[hidden_states.shape[0]] * group.world_size,
        group=group.device_group,
    )
    return gathered, sizes


def scatter_lmhead_logits(logits: torch.Tensor, sizes: list[int], group: "GroupCoordinator") -> torch.Tensor:
    """Return this rank's rows with the vocabulary shards in rank order."""
    if sum(sizes) == 0:
        return logits.new_empty((0, group.world_size * logits.shape[1]))
    if all(size == sizes[0] for size in sizes):
        return group.all_to_all(logits)

    local_size = sizes[group.rank_in_group]
    shard_size = logits.shape[1]
    received = logits.new_empty((group.world_size * local_size, shard_size))
    dist.all_to_all_single(
        received,
        logits.contiguous(),
        output_split_sizes=[local_size] * group.world_size,
        input_split_sizes=sizes,
        group=group.device_group,
    )
    # Receive order is [source vocabulary shard, local token, vocabulary].
    return (
        received.view(group.world_size, local_size, shard_size)
        .transpose(0, 1)
        .reshape(local_size, group.world_size * shard_size)
    )
