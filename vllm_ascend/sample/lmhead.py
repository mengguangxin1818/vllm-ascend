# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.sampling_params import SamplingParams


def get_lmhead_candidate_count(sampling_params: Iterable["SamplingParams"]) -> int:
    """Return a safe local candidate count, or zero if full logits are needed.

    Constraints that can change the vocabulary ranking must run before
    truncation. Leave these requests to the existing local sampling pipeline.
    """
    candidate_count = 1
    for params in sampling_params:
        if (
            params.presence_penalty != 0
            or params.frequency_penalty != 0
            or params.repetition_penalty != 1
            or params.min_tokens
            or params.min_p
            or params.logit_bias
            or params.allowed_token_ids is not None
            or params.bad_words
            or params.structured_outputs is not None
            or params.logprobs is not None
            or params.prompt_logprobs is not None
            or params.logprob_token_ids
            or params.thinking_token_budget is not None
            or params.extra_args
        ):
            return 0
        if params.temperature == 0:
            continue
        if params.top_k <= 0:
            return 0
        candidate_count = max(candidate_count, params.top_k)
    return candidate_count


def synchronize_lmhead_candidate_count(candidate_count: int, group: "GroupCoordinator") -> int:
    """Agree on the exchange shape before both real and dummy logits calls.

    These are CPU configuration values, so this does not read back NPU data.
    Idle ranks contribute one candidate and still join the logits collectives.
    """
    requirements = torch.tensor([candidate_count == 0, candidate_count], dtype=torch.int64, device="cpu")
    if group.world_size > 1:
        dist.all_reduce(requirements, op=dist.ReduceOp.MAX, group=group.cpu_group)
    requires_full_logits, max_candidates = requirements.tolist()
    return 0 if requires_full_logits else max_candidates


def reduce_lmhead_logits(
    logits: torch.Tensor,
    candidate_count: int,
    vocab_start: int,
    num_valid_tokens: int,
    vocab_size: int,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Exchange vocabulary candidates and return only this DP rank's rows.

    Input rows contain the padded batches of every LM head rank. Exchange
    candidates along the row dimension, then reconstruct local vocabulary
    logits so request metadata and random generators remain entirely local.
    """
    logits = logits.clone()
    logits[:, num_valid_tokens:] = -float("inf")
    count = min(candidate_count, logits.shape[-1])
    if count == 1:
        # max preserves the first vocabulary index when logits tie.
        values, indices = logits.max(dim=-1, keepdim=True)
    else:
        values, indices = logits.topk(count, dim=-1)
        # A group can contain both greedy and top-k requests. Keep the first
        # local maximum even when topk chooses a different subset of ties,
        # so reconstructing logits preserves torch.argmax's tie breaking.
        first_max = logits.argmax(dim=-1, keepdim=True)
        has_first_max = (indices == first_max).any(dim=-1, keepdim=True)
        indices[:, :1] = torch.where(has_first_max, indices[:, :1], first_max)
    indices = indices + vocab_start
    values = group.all_to_all(values)
    indices = group.all_to_all(indices)
    # Use an extra column for vocabulary padding, including empty shards.
    indices = indices.clamp(max=vocab_size)
    local_logits = logits.new_full((values.shape[0], vocab_size + 1), -float("inf"))
    local_logits.scatter_(1, indices, values)
    return local_logits[:, :vocab_size]
