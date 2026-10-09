# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Four-NPU validation, including HCCL zero splits and changing batch lengths."""

import tempfile

import torch
import torch.multiprocessing as mp

from tests.ut.distributed.test_lmhead_communication import run_exchange_worker


def test_lmhead_alltoallv_hccl():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(run_exchange_worker, args=(f"file://{directory}/rendezvous", "hccl"), nprocs=4, join=True)
    torch.npu.synchronize()
