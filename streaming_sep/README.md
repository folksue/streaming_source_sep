# Streaming EnCodec Vocal Separation Baseline

This is a standalone baseline for streaming vocal/source separation in the current project root.
It does not modify `third_party/streaming_svs_prototype`.

## Design

- Codec: causal `facebook/encodec_24khz`.
- Task: predict source EnCodec RVQ codes from mixture EnCodec RVQ codes.
- Main stream: autoregressively predicts source RVQ layer 0.
- Residual stream: predicts RVQ layers `1..K-1` from the same hidden state.
- Next-step source feedback uses the summed embeddings of the previous source frame's full RVQ code stack, matching the Qwen3-TTS pattern more closely than feeding only codebook 0.
- Prompt hook: `audio_prompt_codes` is accepted by the model and inference script, but defaults to no prompt.

## Manifest

Each manifest item is a JSON object:

```json
{
  "utt_id": "song_0001",
  "mixture_path": "audio/song_0001_mix.wav",
  "source_path": "audio/song_0001_vocal.wav",
  "audio_prompt_path": "audio/optional_prompt.wav"
}
```

`audio_prompt_path` is optional.

## Commands

Install dependencies in your preferred environment:

```bash
pip install -r requirements-sep.txt
```

Build EnCodec caches:

```bash
python -m streaming_sep.preprocess_encodec --config configs/encodec_sep_200m.yaml --split both
```

Train:

```bash
python -m streaming_sep.train --config configs/encodec_sep_200m.yaml
```

Train directly from the handoff EnCodec shard indices:

```bash
python -m streaming_sep.train_handoff_indexes \
  --config configs/encodec_sep_200m.yaml \
  --data-root /cfs4/folkswei/test/ft_local/streaming_sep_offload \
  --residual-predictor mtp
```

`--residual-predictor mtp` uses a small causal Transformer over RVQ codebook positions within each frame.
Use `--residual-predictor parallel` to fall back to the older independent residual heads.

Run inference:

```bash
python -m streaming_sep.infer_stream \
  --checkpoint artifacts/sep/checkpoints/epoch_0001.pt \
  --mixture path/to/mixture.wav \
  --stem vocal \
  --out artifacts/sep/outputs/vocal.wav
```

## Notes

The current `generate_stream` interface emits one EnCodec frame of source codes at a time.
It recomputes the prefix for simplicity; the interface is already compatible with replacing that path by KV-cache decoding later.
