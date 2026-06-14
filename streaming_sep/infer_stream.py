from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from streaming_sep.codec import CausalEncodecTokenizer
from streaming_sep.train import build_model


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mixture", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--audio-prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sample_rate = int(cfg["audio"]["sample_rate"])

    tokenizer = CausalEncodecTokenizer(
        model_name=str(cfg["audio"].get("encodec_model_name", "facebook/encodec_24khz")),
        bandwidth=cfg["audio"].get("encodec_bandwidth", 6.0),
        device=device,
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    mixture_codes = tokenizer.encode(load_audio(args.mixture, sample_rate), sample_rate).codes.to(device)
    prompt_codes = None
    prompt_mask = None
    if args.audio_prompt:
        prompt_codes = tokenizer.encode(load_audio(args.audio_prompt, sample_rate), sample_rate).codes.to(device)
        prompt_codes = prompt_codes.unsqueeze(0)
        prompt_mask = torch.ones(prompt_codes.size(0), prompt_codes.size(1), dtype=torch.bool, device=device)

    generated = []
    for codes in model.generate_stream(
        mixture_codes=mixture_codes.unsqueeze(0),
        audio_prompt_codes=prompt_codes,
        audio_prompt_mask=prompt_mask,
        temperature=args.temperature,
    ):
        generated.append(codes.squeeze(0).cpu())

    source_codes = torch.stack(generated, dim=0)
    wav = tokenizer.decode(source_codes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav.numpy(), sample_rate)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

