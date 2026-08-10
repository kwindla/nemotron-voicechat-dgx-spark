# Current-work preservation and public-main port plan

Date: 2026-08-10

## Objective

Preserve the complete current implementation on a temporary Git branch, then
port the reviewed changes onto the current public `main` without losing the
already-promoted `codex/promotion-assembly` behavior. In particular, the port
must retain the qualified 160 ms Pipecat output prebuffer that is absent from
the current working branch.

The currently running model/Pipecat/ngrok stack remains untouched until a
reviewed replacement is ready for manual testing.

## Known starting state

- Current branch: `codex/post-fc-pair-recovery`
- Current base commit: `e19f906547f32156bbbc872d16142dca14dd259d`
- Public `main` observed on 2026-08-10:
  `a7cb5c22733002c40c52a1c2db23a37c53e44404`
- Merged promotion tip:
  `b5a023b2932b5ab7c9dac2de8142a4465afbecec`
- The current branch and promotion branch do not share usable Git ancestry.
- `old-env` is an untracked machine-local environment file containing real
  credentials and paths. It must never be staged or copied into the port.
- `proj-2026-08-08-1515/` contains local planning and agent-session artifacts;
  it is not product source and must not be staged or copied into the port.
- The running public-runtime image at plan review is
  `sha256:883c4220db9ac93b85b0e73dffba66ad21459b86f24140a17e796ca46d40fe2c`.
  Its source, payload, and recipe labels must be recorded again immediately
  before the preservation commit rather than assumed from this observation.

## Phase 1: preserve the current tree

1. Record the starting branch, commit, tracked binary diff hash, complete
   untracked-file inventory, running image ID, and the image's exact project
   source/payload/recipe identity labels outside Git.
2. Scan the files intended for staging for credentials, retained audio, model
   weights, caches, virtual environments, and other machine-local artifacts.
3. Create temporary branch `codex/current-work-snapshot-20260810` at the
   current commit.
4. Stage an explicit allowlist of project source, tests, tools, configuration,
   container recipes, documentation, licenses, and the two small retained JSON
   evidence records. Do not use an indiscriminate staging command.
5. Add `old-env` and `proj-*/` to `.gitignore`, then exclude those paths along
   with `.env`, caches, virtual environments, runtime traces, generated audio,
   model weights, and build outputs. The ignore rules are part of the snapshot
   so a later broad staging command cannot silently publish the current local
   credentials or session artifacts.
6. Commit the staged tree as one preservation snapshot. A single snapshot is
   intentional: it guarantees that cross-cutting work is retained before we
   start reorganizing it.
7. Give every path in the recorded pre-commit modified/untracked inventory one
   explicit disposition: staged, or excluded with a named reason. After the
   commit, compare the committed paths and remaining/ignored paths against that
   complete classification. Record the snapshot commit ID and tree ID.
8. Check out the snapshot commit into a separate clean worktree and recompute
   `public_runtime_source_sha256`, `public_runtime_payload_sha256`, and
   `public_runtime_recipe_sha256` there. They must exactly match the labels of
   the running deployed image recorded immediately before the commit. This is
   the strongest check that the allowlist did not omit any runtime build input.

## Phase 2: establish the public-main port

1. Fetch public `main` and verify that its immutable commit is the expected
   merge descendant of `b5a023b2932b5ab7c9dac2de8142a4465afbecec`.
2. Create a separate clean worktree and branch
   `codex/public-main-port-20260810` from the fetched public-main commit. Do not
   rewrite or clean the preservation worktree.
3. Produce a three-way inventory:
   public main, the preservation snapshot, and the current live runtime source
   identity. Classify every path as promoted baseline, later reviewed work,
   local-only artifact, or obsolete/superseded implementation.
   Promotion-only paths absent from the snapshot must be named and classified
   `promoted baseline - keep`, including all `step7_*` and `step9_*` tools and
   reports, `patch_pair_full_graph*`, EarTTS exact-constants images/patches/
   gates, promotion reports and cache-provenance records, and their tests.
4. Port changes in reviewable dependency order:

   - Runtime/container identity, bootstrap, and immutable build/audit support.
   - Protocol, server, Speech patch, direct-text transactions, function-call
     recovery, EarTTS state handling, and latency optimizations.
   - Pipecat Smart Turn lifecycle, typed-input adapter, qualified 160 ms output
     prebuffer, and bounded unassigned-input buffering.
   - English Nemotron ASR evaluator, fixtures, qualification runners,
     comparators, and regression tests.
   - Documentation, review records, and retained evidence.

5. Preserve public-main behavior by default when the snapshot contains an
   older or missing implementation. The 160 ms output prebuffer is an explicit
   required example. Resolve conflicts semantically; do not accept wholesale
   snapshot overwrites of public-main files.
6. Make focused commits for the port. Each commit must state its dependency and
   verification status. Run the relevant CPU/static tests after each slice,
   then the combined suite, formatting/lint checks, public-runtime identity
   tests, patch applicability checks, and bootstrap/audit checks.

## Phase 3: adversarial review and qualification

1. Give Fable the plan before creating the snapshot commit. Resolve every
   blocking finding and ask Fable to re-review.
2. After the port and tests are complete, give Fable the exact
   `public-main..port` diff, commit list, test results, source-identity results,
   and known residual risks. Resolve blockers and repeat review until approved.
3. Make a literal fresh clone of the port branch into a new directory and
   rebuild through its normal `./voicechat bootstrap` path. Do not reuse a
   working tree whose ignored files, caches, or local artifacts could hide a
   missing committed input.
4. Run the established live qualification gates before replacing the currently
   running stack. This includes voice and typed input, tool calling, independent
   response-audio ASR, retained-audio checks, and a longer multi-turn latency
   probe.

## Phase 4: manual browser handoff

1. Deploy the reviewed runtime, Pipecat service, and ngrok tunnel.
2. Run an automated end-to-end browser test covering microphone input, typed
   input, a real tool call, audio playout continuity, response completion, and
   multiple later turns.
3. Confirm from correlated browser, Pipecat, protocol, and model traces that:

   - the first user words are retained;
   - the 160 ms output prebuffer is active;
   - model text and independently transcribed response audio both complete;
   - a tool request is committed and completes exactly once;
   - later-turn commit and response latency stay within the preregistered bound;
   - there are no overlapping client input turns or fatal protocol errors.

4. Leave the qualified stack and ngrok tunnel running for the user's final
   manual test.

## Stop conditions

- Never delete, reset, or clean the original working tree.
- Do not replace the live stack with an unreviewed or unaudited image.
- Stop the port if public `main` is not the expected promotion descendant.
- Treat any missing snapshot path, unexplained semantic conflict, source-
  identity mismatch, failed qualification gate, or Fable blocker as a release
  blocker rather than silently dropping or weakening it.

## Known port follow-up

The current `container/build-asr-evaluator.sh` lacks the host-network build
workaround used by the public-runtime builders on this DGX. Review and test the
smallest reproducible fix during the port so a fresh-host evaluator build does
not fail on Docker bridge DNS. This is port work, not a reason to weaken the
snapshot identity or evaluator audit.
