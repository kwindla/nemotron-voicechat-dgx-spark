# LibriSpeech ASR canary notice

The independent ASR evaluator image embeds one unmodified LibriSpeech audio clip
(`1272-128104-0000.flac`) solely as a non-silent decoder health check.

- Source corpus: LibriSpeech, OpenSLR SLR12
- Authors: Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Source: https://www.openslr.org/12
- License text: https://creativecommons.org/licenses/by/4.0/legalcode
- Retrieved through the pinned Hugging Face test fixture revision recorded in
  `config/nemotron-asr-en-0.6b.json`

No modification is made to the extracted FLAC bytes. Its SHA-256 and source
container SHA-256 are verified during the image build and audit.
