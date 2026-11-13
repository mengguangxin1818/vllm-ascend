# SPDX-License-Identifier: Apache-2.0
"""Group-agreed, single-use first-draft DP metadata reuse."""

from enum import IntEnum


class MTPDPSyncRole(IntEnum):
    UNSUPPORTED = 0
    ACTIVE = 1
    DUMMY = 2


class MTPDPSyncReuse:
    # Main token/mode, first-draft prediction, role and per-rank consent.
    MAIN_METADATA_ROWS = 5

    def __init__(self):
        self.role = MTPDPSyncRole.UNSUPPORTED
        self.predicted_tokens = 0
        self.pending: tuple[int, ...] | None = None

    def begin_step(self):
        # Invalidate even on an early return, so no prediction crosses rounds.
        self.pending = None
        self.unsupported()

    def unsupported(self):
        self.role = MTPDPSyncRole.UNSUPPORTED
        self.predicted_tokens = 0

    def predict_active(self, tokens):
        if tokens <= 0:
            self.unsupported()
        else:
            self.role = MTPDPSyncRole.ACTIVE
            self.predicted_tokens = int(tokens)

    def local_fields(self):
        return int(self.predicted_tokens), int(self.role)

    def main_completed(self, *, full_graph, predicted_tokens, roles, main_padded_tokens, reuse_flags):
        # A dummy's draft argument is the final main padding, not raw tokens.
        tokens = tuple(
            int(main_padded_tokens[i]) if role == MTPDPSyncRole.DUMMY else int(predicted_tokens[i])
            for i, role in enumerate(roles)
        )
        # Only the FULL-graph four-token bucket has been validated on hardware.
        # This decision uses the completed main collective on every rank.
        self.pending = (
            tokens
            if (
                full_graph
                and all(role in (MTPDPSyncRole.ACTIVE, MTPDPSyncRole.DUMMY) for role in roles)
                and all(n == 4 for n in tokens)
                and all(n == 4 for n in main_padded_tokens)
                and all(flag == 1 for flag in reuse_flags)
            )
            else None
        )

    def clear_prediction(self):
        self.pending = None

    def consume_prediction(self, *, rank, local_tokens, mode):
        tokens, self.pending = self.pending, None
        if tokens is None:
            return None
        # Never enter a rank-local fallback collective after the group agreed
        # to skip: peers would skip and this rank would wait indefinitely.
        if int(local_tokens) != tokens[rank] or int(mode) != 0:
            raise RuntimeError(
                "MTP DP sync reuse contract violated: "
                f"rank={rank}, local_tokens={local_tokens}, predicted={tokens}, mode={mode}. "
                "Disable VLLM_ASCEND_MERGE_MTP_DP_SYNC on all DP workers and restart."
            )
        count = max(tokens)
        return count, (count,) * len(tokens)
