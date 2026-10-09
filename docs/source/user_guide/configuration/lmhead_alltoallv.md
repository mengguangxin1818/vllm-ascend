# Experimental variable-size LMHead exchange

Merge these settings into `--additional-config`:

```json
{
  "finegrained_tp_config": {"lmhead_tensor_parallel_size": 4},
  "enable_reduce_sample": false,
  "enable_lmhead_alltoallv": true
}
```

The option defaults to false. It activates separately for target post-forward
logits and eager MTP logits, only when the existing DP metadata synchronization
is skipped. The initial implementation requires PP=PCP=DCP=1 and no LoRA.
Other configurations retain the existing padding path. Combining this option
with reduce sampling emits a warning and retains the existing reduce-sample
path; variable-size LMHead exchange remains inactive. Startup logs report
activation for each model.

Each LMHead call performs four phases:

1. AllGather one int32 valid row count per rank on the LMHead CPU group.
2. Replicate local hidden states for each destination and collect them using
   AllToAllV, yielding `[sum(row_counts), hidden_size]` on every rank.
3. Compute logits for the local vocabulary shard.
4. AllToAllV the logits back to their owning ranks, then concatenate vocabulary
   shards in rank order. This reuses the existing `group.all_to_all` wrapper with
   `scatter_sizes=row_counts` and `gather_sizes=[local_vocab_size] * group_size`.
   Existing sampling receives local full-vocabulary logits.

Equal nonempty batches use ordinary AllGather and AllToAll. All-empty groups
skip data exchange and GEMM after the length collective. Target dummy calls
contribute zero rows; eager MTP dummy calls retain the synthetic sampling rows
needed by the draft loop. Each MTP step exchanges its own row counts; cross-step
reuse is not implemented. Backbone inputs and DP synchronization are unchanged.

## Validation before deployment

The CPU tests run real four-process Gloo exchanges for equal, unequal, idle,
all-empty, and changing batches, in FP32 and BF16. The isolated logits processor
method is compared with the original padded path, including bias and vocabulary
trimming. These tests do not load the complete model runner or validate ACLGraph.

Run the standalone CPU suite in an environment with CPU PyTorch:

```bash
GLOO_SOCKET_IFNAME=lo python -m unittest discover \
  -s tests/ut/distributed -p test_lmhead_communication.py -v
```

On a host with four A3 NPUs and the project dependencies installed, run:

```bash
pytest -v tests/ut/distributed/a3_4/test_lmhead_alltoallv.py
```

The hardware test reuses the exchange and logits comparisons with HCCL device
communication and a Gloo CPU metadata group. It includes zero send/receive splits
and consecutive changing batches, and exercises the existing NPU list-based
AllToAll wrapper for logits. The CPU adapter uses Gloo's `all_to_all_single`
because Gloo does not support list-based AllToAll. It is a communication test,
not a model ITL benchmark.

This experimental path has not been validated on NPU hardware. Verify zero
splits with the deployed torch_npu/HCCL version. Compare valid logits with the
default path for equal, uneven, idle, and all-empty ranks, including mixed
real/dummy execution and every MTP step. Confirm that disabling the option
preserves the existing behavior.

Measure end-to-end ITL as well as length synchronization, input replication,
both data collectives, GEMM, and output layout conversion. Reduced padding does
not guarantee lower latency. Compare against both fixed padding and existing
dynamic padding, and confirm the startup activation message before benchmarking.
