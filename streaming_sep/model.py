from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SepModelOutput:
    coarse_logits: torch.Tensor  # [B, T, V]
    residual_logits: torch.Tensor  # [B, T, K-1, V]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def _rope_cache(seq_len: int, dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [T, D/2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos.view(1, 1, cos.size(0), cos.size(1))
    sin = sin.view(1, 1, sin.size(0), sin.size(1))
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, rope_base: float = 10000.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None, sliding_window: int | None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = _rope_cache(seq_len, self.head_dim, x.device, self.rope_base)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        blocked = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
        if sliding_window is not None:
            idx = torch.arange(seq_len, device=x.device)
            too_old = (idx[:, None] - idx[None, :]) >= int(sliding_window)
            blocked = blocked | too_old
        if key_padding_mask is not None:
            blocked = blocked.unsqueeze(0).unsqueeze(0) | key_padding_mask[:, None, None, :]
        else:
            blocked = blocked.unsqueeze(0).unsqueeze(0)
        attn_mask = torch.zeros_like(blocked, dtype=x.dtype).masked_fill(blocked, torch.finfo(x.dtype).min)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None, sliding_window: int | None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask, sliding_window=sliding_window)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DualStreamEncodecSeparator(nn.Module):
    """Qwen3-TTS style coarse/residual RVQ predictor for source separation."""

    def __init__(
        self,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        d_model: int = 1024,
        num_layers: int = 18,
        num_heads: int = 16,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        sliding_window: int | None = 1024,
        prompt_dim: int = 1024,
    ) -> None:
        super().__init__()
        if num_codebooks < 2:
            raise ValueError("RVQ separation baseline expects at least 2 codebooks")
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.d_model = int(d_model)
        self.sliding_window = None if sliding_window is None else int(sliding_window)

        self.mix_code_emb = nn.ModuleList([nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)])
        self.source_code_emb = nn.ModuleList([nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)])
        self.source_bos = nn.Parameter(torch.zeros(d_model))
        self.prompt_code_emb = nn.ModuleList([nn.Embedding(codebook_size, prompt_dim) for _ in range(num_codebooks)])
        self.prompt_proj = nn.Linear(prompt_dim, d_model)
        self.input_norm = RMSNorm(d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ffn_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_norm = RMSNorm(d_model)
        self.coarse_head = nn.Linear(d_model, codebook_size, bias=False)
        self.residual_heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size, bias=False) for _ in range(num_codebooks - 1)]
        )

    def _embed_rvq(self, embeddings: nn.ModuleList, codes: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for idx, emb in enumerate(embeddings):
            out = out + emb(codes[..., idx])
        return out / math.sqrt(len(embeddings))

    def _shift_source_codes(self, source_codes: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = source_codes.shape
        bos = self.source_bos.view(1, 1, -1).expand(bsz, 1, -1)
        if seq_len == 1:
            return bos
        prev = self._embed_rvq(self.source_code_emb, source_codes[:, :-1])
        return torch.cat([bos, prev], dim=1)

    def _prompt_context(self, audio_prompt_codes: torch.Tensor | None, audio_prompt_mask: torch.Tensor | None) -> torch.Tensor | None:
        if audio_prompt_codes is None:
            return None
        prompt = self._embed_rvq(self.prompt_code_emb, audio_prompt_codes)
        if audio_prompt_mask is None:
            pooled = prompt.mean(dim=1)
        else:
            mask = audio_prompt_mask.to(prompt.dtype).unsqueeze(-1)
            pooled = (prompt * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.prompt_proj(pooled).unsqueeze(1)

    def forward(
        self,
        mixture_codes: torch.Tensor,
        source_codes: torch.Tensor,
        mask: torch.Tensor | None = None,
        audio_prompt_codes: torch.Tensor | None = None,
        audio_prompt_mask: torch.Tensor | None = None,
    ) -> SepModelOutput:
        mix = self._embed_rvq(self.mix_code_emb, mixture_codes)
        src_prev = self._shift_source_codes(source_codes)
        prompt = self._prompt_context(audio_prompt_codes, audio_prompt_mask)
        x = mix + src_prev
        if prompt is not None:
            x = x + prompt
        x = self.input_norm(x)

        key_padding_mask = None if mask is None else ~mask.bool()
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, sliding_window=self.sliding_window)
        x = self.out_norm(x)

        coarse_logits = self.coarse_head(x)
        residual_logits = torch.stack([head(x) for head in self.residual_heads], dim=2)
        return SepModelOutput(coarse_logits=coarse_logits, residual_logits=residual_logits)

    @torch.no_grad()
    def generate_stream(
        self,
        mixture_codes: torch.Tensor,
        audio_prompt_codes: torch.Tensor | None = None,
        audio_prompt_mask: torch.Tensor | None = None,
        temperature: float = 0.0,
    ):
        source_prefix = torch.zeros(
            mixture_codes.size(0),
            0,
            self.num_codebooks,
            dtype=torch.long,
            device=mixture_codes.device,
        )
        for step in range(mixture_codes.size(1)):
            dummy_next = torch.zeros(
                mixture_codes.size(0),
                1,
                self.num_codebooks,
                dtype=torch.long,
                device=mixture_codes.device,
            )
            source_in = torch.cat([source_prefix, dummy_next], dim=1)
            out = self(
                mixture_codes=mixture_codes[:, : step + 1],
                source_codes=source_in,
                mask=torch.ones(mixture_codes.size(0), step + 1, dtype=torch.bool, device=mixture_codes.device),
                audio_prompt_codes=audio_prompt_codes,
                audio_prompt_mask=audio_prompt_mask,
            )
            coarse = sample_logits(out.coarse_logits[:, -1], temperature)
            residual = sample_logits(out.residual_logits[:, -1], temperature)
            next_codes = torch.cat([coarse[:, None], residual], dim=1)
            source_prefix = torch.cat([source_prefix, next_codes[:, None, :]], dim=1)
            yield next_codes


def sample_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature and temperature > 0:
        probs = torch.softmax(logits / float(temperature), dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()
    return logits.argmax(dim=-1)


def separation_loss(
    output: SepModelOutput,
    source_codes: torch.Tensor,
    mask: torch.Tensor,
    residual_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    coarse_loss = F.cross_entropy(
        output.coarse_logits.reshape(-1, output.coarse_logits.size(-1)),
        source_codes[..., 0].reshape(-1),
        reduction="none",
    ).view_as(mask)
    coarse_loss = (coarse_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    residual_target = source_codes[..., 1:]
    residual_loss = F.cross_entropy(
        output.residual_logits.reshape(-1, output.residual_logits.size(-1)),
        residual_target.reshape(-1),
        reduction="none",
    ).view(mask.size(0), mask.size(1), -1)
    residual_loss = (residual_loss * mask.float().unsqueeze(-1)).sum() / (
        mask.float().sum().clamp_min(1.0) * residual_target.size(-1)
    )

    loss = coarse_loss + float(residual_weight) * residual_loss
    with torch.no_grad():
        coarse_acc = ((output.coarse_logits.argmax(dim=-1) == source_codes[..., 0]) & mask).float().sum()
        coarse_acc = coarse_acc / mask.float().sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "coarse_loss": float(coarse_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "coarse_acc": float(coarse_acc.detach().cpu()),
    }
