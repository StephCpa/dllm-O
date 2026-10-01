# Phase 2 Option B Smoke Audit and Pre-Launch Amendment

**Audit date:** 2026-09-03  
**Status:** Formal Phase 2 launch remains blocked pending a corrected freeze and
repeat smoke. No Phase 2 correctness outcome has been read.

## Finding

The four planned smoke cells each produced 50 atomic result/trace pairs, for
200 pairs total, with no temporary files left behind. However, every runner
invocation failed while validating its completion marker. The Phase 2 schema
admitted `phase2-extension` for generation records but accidentally omitted
that same run kind from the `shard_completion.run_kind` enumeration.

The smoke wrapper continued through all four cells and printed `rc=0` despite
the marker-validation tracebacks. Because the wrapper used only `set -u`, its
completion banner is not accepted as evidence of success. Future launch
control must use `set -euo pipefail`, capture the runner status explicitly,
and require the expected marker before advancing to another cell or shard.

The smoke therefore did **not** pass its artifact contract. The 200 generated
pairs are preserved for audit but will be isolated from the formal output tree
before a repeat smoke under the corrected freeze. They will not be reused by
the formal 19,200-generation run.

## Mechanical Amendment

The only schema change is to add the already frozen run kind
`phase2-extension` to the completion-marker enumeration. A regression test now
constructs a marker through the production runner and validates it against the
Phase 2 schema. The Phase 2 freeze will be re-emitted against the corrected
schema and new runner/analyzer commit, with the original freeze SHA
`6c97f7a82bd97d2b9b1d1e1da2b44729c9c980075090c944c6590119a805e40c`
recorded as superseded.

This amendment changes no task, query, policy, seed, rollout count, decoder
parameter, endpoint, bootstrap procedure, verdict taxonomy, or Condition A
gate. It precedes the formal run and any Phase 2 correctness read.

## Smoke Resource Measurements

The four 50-generation cells took 27:45, 25:23, 23:45, and 24:00,
respectively. This is approximately 30.3 seconds per generation on average,
about 6% slower than the Phase 1 planning baseline of 28.61 seconds. It remains
below the frozen 10% logging-overhead gate, so the top-64 logging design is
retained rather than reduced to top-32.

The 200 compressed full-distribution traces occupy 78,569,376 bytes, or a mean
of 392,847 bytes per generation. At 19,200 generations this projects to roughly
7.54 GB of compressed traces. The server had about 819 GB free at audit time,
so storage does not block the formal run.

## Acceptance Requirements for the Repeat Smoke

1. All four cells return zero under a fail-fast wrapper.
2. Each cell has exactly 50 result/trace pairs and no temporary file.
3. Each trace hash and record provenance validates under the corrected schema.
4. Each completion marker validates, exists at the expected path, and has
   identical expected/observed key hashes.
5. The corrected Phase 2 freeze SHA and runner commit appear in every record.
6. No correctness field is read during smoke acceptance.

