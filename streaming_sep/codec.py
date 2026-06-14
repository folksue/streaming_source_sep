from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class EncodecCodes:
    codes: torch.Tensor  # [T, K]
    sample_rate: int
    bandwidth: float | None


class CausalEncodecTokenizer:
    """Thin wrapper around the causal 24 kHz EnCodec checkpoint."""

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        bandwidth: float | None = 6.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if "48khz" in model_name.lower():
            raise ValueError("Use the 24 kHz causal EnCodec checkpoint for streaming.")
        from transformers import AutoProcessor, EncodecModel  # type: ignore

        self.device = torch.device(device)
        self.model_name = model_name
        self.bandwidth = bandwidth
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if bandwidth is not None and hasattr(self.model, "set_target_bandwidth"):
            self.model.set_target_bandwidth(float(bandwidth))
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.processor, "sampling_rate", 24000))

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, sample_rate: int) -> EncodecCodes:
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate}. Resample before encoding.")
        wav = wav.detach().float().cpu()
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        if wav.dim() != 1:
            raise ValueError(f"Expected mono waveform [N] or [C,N], got shape={tuple(wav.shape)}")
        inputs = self.processor(
            raw_audio=wav.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        encoded = self.model.encode(**inputs)
        audio_codes = encoded.audio_codes
        if audio_codes.dim() == 4:
            # HF layout is usually [B, chunks, K, T].
            audio_codes = audio_codes.reshape(audio_codes.size(0), audio_codes.size(2), -1)
        if audio_codes.dim() != 3:
            raise RuntimeError(f"Unexpected EnCodec code shape: {tuple(audio_codes.shape)}")
        codes = audio_codes[0].transpose(0, 1).contiguous().cpu().long()
        return EncodecCodes(codes=codes, sample_rate=self.sample_rate, bandwidth=self.bandwidth)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 2:
            raise ValueError(f"Expected codes [T,K], got shape={tuple(codes.shape)}")
        audio_codes = codes.transpose(0, 1).unsqueeze(0).to(self.device)
        audio_scales = [None]
        decoded = self.model.decode(audio_codes=audio_codes, audio_scales=audio_scales)
        wav = decoded.audio_values if hasattr(decoded, "audio_values") else decoded[0]
        return wav.squeeze().detach().cpu()

    @torch.inference_mode()
    def decode_stream(self, code_chunks: Iterable[torch.Tensor]) -> torch.Tensor:
        wavs = [self.decode(chunk) for chunk in code_chunks]
        if not wavs:
            return torch.empty(0)
        return torch.cat(wavs, dim=-1)

