# Weight reproduction status

The normal installation path downloads the signed Production Candidate 1 Nano
and EarTTS derivatives from the immutable Hugging Face revision recorded in
`config/artifact-release.json`. Bootstrap verifies the complete published
inventory and release signature before the runtime can use it.

The initial public repository release contains the consumption-side Nano and
EarTTS component gates, artifact verification, and the complete downloaded-
artifact end-to-end qualification suite. It does **not** yet contain the
production conversion pipeline or claim that a public source-conversion run has
been completed.

The deterministic conversion pipeline will be published in a stage-2 follow-up
after two clean conversions have reproduced the published bytes and the
locally converted artifacts have passed the same live suite. That follow-up
will include the exact conversion command, corpus-selection tooling, disk
guards, resumable stage markers, and dual-source comparison evidence. Until
then, the corresponding item in `docs/release-checklist.md` intentionally
remains unchecked.

The provenance statements and recipes already present in this repository
describe how the published weights were produced; they are retained as true
historical provenance. They are not a claim that the stage-2 executable
pipeline ships in this initial commit.
