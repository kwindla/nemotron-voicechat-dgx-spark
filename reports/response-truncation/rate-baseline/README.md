# Production truncation baseline

Measured on the production build (`session-repair-20260818-r4`, frozen
contract) so that any candidate fix has a number to beat. Counts, not an
estimated production rate: three replay sessions of one retained conversation.

| metric | value |
|---|---:|
| completed responses | 7 |
| truncated | 1 (14.3%) |
| long responses (>= 30 generated words) | 3 |
| long responses truncated | 1 (33%) |

The reproduced loss was a 57-word story that spoke 41 words and dropped its
final 16: "from then on they were inseparable exploring the world together and
spreading joy wherever they went". Same signature as every other observed
loss - a long response losing its tail.

Truncation is confirmed by pinned offline ASR over the emitted audio, diffed
against the generated text. Responses the user (or replay) interrupted are
excluded, since a short interrupted response is not a defect.

Note on power: EarTTS acoustic sampling is stochastic with no per-response
seed, so incidence varies run to run and a single campaign cannot establish a
rate. At roughly 33% incidence on long responses, about 10 long responses per
arm are needed before a clean zero is meaningful.

Rebuild with `python3 reports/response-truncation/truncation_rate.py --sessions N`
against a running bot with playout tracing enabled.
