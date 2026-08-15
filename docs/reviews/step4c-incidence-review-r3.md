# Step 4c incidence measurement-validity review — round 3

**Verdict: PASS — Step 4c is admissible (`INSUFFICIENT`, pivot standing);
Step 4d may proceed.** The single round-2 delta is closed. The public analyzer
now fails closed on the preregistered paced-source duration evidence, the new
end-to-end cases exercise the missing and truncated forms without manufacturing
compliant fixture values, and the retained result and matched-state record
reproduce byte for byte.

This review was entirely offline. I made no health query, model connection,
browser connection, live probe, container change, or commit. Apart from this
review file, I made no repository change.

## Round-2 delta closure

The source-duration gate is now evidence-enforced rather than merely declared:

- The immutable contract derives exactly 1,500 frames from the registered
  120.0-second duration and 80 ms pacing
  (`tools/qualification/step4c_contract.py:18-26`).
- Direct and native Pipecat reports must contain an integer, non-boolean
  `source_frames` equal to that expected count. The check is part of the
  ordinary explicit-timing predicate, and block 01's legacy timing-sidecar path
  calls the same check (`tools/qualification/step4c_incidence.py:170-175,179-214,233-245`).
- A potentially valid browser block must retain a finite numeric
  `session.connected_window_seconds` covering duration plus drain; the only
  allowance is the separately stated 1 ms clock-edge tolerance
  (`tools/qualification/step4c_incidence.py:351-361`).

I reran the exact retained-evidence mutation that failed round 2. In a temporary
copy, I changed only valid direct block 06's `source_frames` from 1,500 to 1 and
pointed the copied manifest at that report. The public CLI exited nonzero with:

```text
ValueError: block 6 is declared valid but evidence fails: ['FIXTURE_TIMING_MISMATCH']
```

The fixtures do not normalize this gate away. The synthetic direct and Pipecat
builders explicitly write `EXPECTED_SOURCE_FRAMES`, and the browser builder
explicitly writes the 135-second connected window
(`tests/runtime/test_step4c_incidence.py:171-180,195-204,225-250`). The public
`analyze()` mutation matrix independently changes each direct/native frame
count to 1, deletes each count, truncates the browser window beyond the allowed
tolerance, and deletes the browser measurement; every case expects
`FIXTURE_TIMING_MISMATCH` (`tests/runtime/test_step4c_incidence.py:454-470,513-546`).
The boundary allowance itself also has a positive test at exactly the stated
1 ms tolerance (`:434-451`).

## No-regression spot-check

Fresh public analysis of the unmodified retained campaign is byte-identical to
both saved outputs. It still yields five valid and four consumed-invalid blocks,
direct 0/458 plus native Pipecat 0/306, browser unavailable, and **0 / 764** valid
`delivered-nonBOS` rows overall. The decision remains `INSUFFICIENT`, and
`matched-states.json` remains `not-triggered`. The narrowed conclusion and the
older-state reconstruction/profiling pivot from round 2 therefore stand; no new
live exposure or Nsight selection is implied.

Fresh offline gates:

- Step 4c public-analyzer end-to-end tests: **29 passed**;
- full suite excluding `browser_e2e`: **1,160 passed, 15 skipped, 3 deselected**
  (four warnings);
- canonical retained verifier: **PASS (67 anchors)**;
- task-scoped Ruff over the recorded Step 4c closure file set: **all checks
  passed**.

No blocking delta remains from round 2. **Step 4d may proceed on the standing
`INSUFFICIENT` result and older-state pivot.**
