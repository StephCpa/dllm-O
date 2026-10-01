# Random-B32 Extension Launch Audit

**Date:** 2026-09-12  
**Evidence status:** post-unblinding-selected targeted prospective follow-up  
**Operative runner commit:** `91fa69d5741709a11eb38a33689f8887cfd42685`  
**Freeze SHA-256:** `5b84f29e1b57415c654bdf377dd775fa1bbe098679215a51c2e46a06e36d44d3`  
**Artifact-schema SHA-256:** `1f9e06530993ea89b972720671e96bdca8bbc348237a0989b44bc2127b8362bf`

## Pre-launch gates

- The full local test suite passed: `108 passed`.
- The targeted runner/analyzer suite passed after the final validation hardening:
  `18 passed`.
- Dry runs returned exactly 3,200 work items for each of three shards and
  9,600 work items in total.
- The expected work-key hashes are:
  - shard 0: `a6618549be2d7d1e121844db3522682ecd00da0df955deadf3755b200b94fdf2`;
  - shard 1: `4401d05f98cef3ab7bd97d9ea5efe8b4fafd925da9e555aafd837db048581074`;
  - shard 2: `02bd47a60857a92d66777ca58de422f5033af10746264c3df20dc4aff9f69c1e`.
- One GSM8K and one MATH-500 canary completed under the operative freeze.
  Their measured generation times were 28.27 and 26.20 seconds.
- Canary validation found 2 valid records, 0 invalid records, and 0 shard
  markers, and logged `reads_random_extension_correctness=false`.

## Corrected pre-launch incident

The first canary commands used a new top-level output root. The prerequisite
guard stopped both commands before generation because the frozen first-16
screening records were not under that root. No result, trace, or shard marker
was written there. The retry correctly used the existing
`outputs/dllm_order_phase1` root; the new `random-extension` run-kind directory
still keeps all extension artifacts disjoint from prior phases. The failed and
successful canary logs are retained under
`outputs/dllm_order_random_extension/random-extension-v1/logs/`.

## Formal launch

At `2026-09-12T22:39:04+08:00`:

- shard 0 started on GPU 3, PID `2573422`;
- shard 1 started on GPU 4, PID `2573428`;
- shard 2 initially remained pending because GPU 1 belonged to another user's
  active high-load process and GPU 5 was reserved for another task.

Both launched processes loaded the pinned local model snapshot and reached
active generation. GPU memory was approximately 18.4 GB per process. The
formal logs are `formal_shard0_gpu3.log` and `formal_shard1_gpu4.log`.

At `2026-09-13T02:22:30+08:00`, the user confirmed that GPU 5 was available.
The device had approximately 46.5 GB free, with only an idle Ollama process
occupying approximately 2 GB. Shard 2 was then launched on GPU 5 as PID
`2583756`, with the same runner commit, freeze, data, model, and command
arguments apart from the frozen shard index and device. It loaded the model,
used approximately 20.4 GB total device memory including Ollama, and completed
its first work item. Its formal log is `formal_shard2_gpu5.log`.

The inference worktree must remain clean and fixed at the operative runner
commit above. Later analysis/documentation commits must not be pulled into that
worktree while inference is active.

## Frozen interpretation boundary

The sole primary endpoint is the paired random-minus-AO contrast in
`HighBudgetCoverage = mean(Pass@16, Pass@32, Pass@64)`, pooled by equal task
weight. The extension cannot revise the original Gate-1 null and cannot support
a universal random-policy or crossover claim. No outcome field is to be read
until all 9,600 new records and all three shard markers pass validation.
