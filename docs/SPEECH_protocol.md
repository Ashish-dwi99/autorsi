# Speech track: AutoRSI on a small speech-to-speech model (protocol, NOT run)

Nothing here was run. The 2GB-RAM box cannot load any candidate.
- LLaMA-Omni2-0.5B-Bilingual (ICTNLP): ~1.93B params (~3.9GB bf16) plus Whisper-large-v3 and a CosyVoice2 decoder.
- SmolDuplex (PranavHarshan): PRD only, no weights.
- Moshi: larger.

## Objectives
1. Reasoning probe (measurable locally on a Mac): spoken-question to ASR transcript, scored on a fixed 50-question GSM8K/ARC subset by exact answer match. Gate: no drop on a held-out 25.
2. Naturalness (partly measurable): UTMOS predicted-MOS on generated audio, plus WER of ASR on the output vs intended text. UTMOS is a proxy, not human judgment; human MOS is not locally evaluable.
3. Not locally evaluable: true full-duplex turn-taking quality and interruption handling (needs live audio loops and human raters).

## Edit space
Only LM-side weights (down_proj of the last layers) via DeltaEdit, keeping speech decoder frozen so the voice cannot be silently damaged; gate on UTMOS and WER too.
Run on a Mac (16GB+): adapt `examples/qwen_coder/` to call the speech pipeline for the gates. This is an untested sketch.
