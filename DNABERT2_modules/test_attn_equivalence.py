"""Numerical equivalence check: FlashAttention (>=2.4) ALiBi path vs the
pure-PyTorch reference path.

The two paths must produce the same outputs (up to fp16 tolerance) because the
flash kernel's `alibi_slopes` convention (`-slope * |i - j|` for self-attention)
is exactly the dense bias the torch path adds. Run on a CUDA box with
`flash-attn>=2.4` installed:

    python -m DNABERT2_modules.test_attn_equivalence

Skips gracefully (exit 0) when CUDA or flash-attn are unavailable.
"""

import sys

import torch

from .configuration_bert import BertConfig
from .bert_layers import BertModel, flash_attn_varlen_qkvpacked_func


def _make_config(attn_impl: str) -> BertConfig:
    return BertConfig(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        attn_impl=attn_impl,
    )


def main() -> int:
    if not torch.cuda.is_available():
        print('SKIP: CUDA not available; the flash path requires a GPU.')
        return 0
    if flash_attn_varlen_qkvpacked_func is None:
        print('SKIP: flash-attn (>=2.4) is not installed.')
        return 0

    torch.manual_seed(0)
    device = 'cuda'

    # Reference model on fp32, flash model on fp16 — identical weights.
    ref = BertModel(_make_config('torch'), add_pooling_layer=False).to(device)
    ref.eval()
    flash = BertModel(_make_config('flash'), add_pooling_layer=False).to(device)
    flash.load_state_dict(ref.state_dict())
    flash = flash.half().eval()

    batch, seqlen = 4, 96
    input_ids = torch.randint(1, 32, (batch, seqlen), device=device)
    # Variable-length sequences so unpadding/cu_seqlens are exercised.
    attention_mask = torch.ones(batch, seqlen, dtype=torch.long, device=device)
    lengths = [96, 80, 64, 33]
    for b, n in enumerate(lengths):
        attention_mask[b, n:] = 0

    with torch.no_grad():
        out_ref = ref(input_ids, attention_mask=attention_mask).last_hidden_state
        out_flash = flash(input_ids,
                          attention_mask=attention_mask).last_hidden_state.float()

    # Compare only real (non-pad) token positions.
    mask = attention_mask.bool().unsqueeze(-1)
    diff = (out_ref - out_flash).abs()
    diff = diff[mask.expand_as(diff)]
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = out_ref[mask.expand_as(out_ref)].abs().mean().item()

    print(f'max_abs_diff={max_abs:.4e}  mean_abs_diff={mean_abs:.4e}  '
          f'ref_scale={ref_scale:.4e}')

    # fp16 path vs fp32 reference: expect agreement to ~1e-2 relative.
    tol = 2e-2 * max(ref_scale, 1.0)
    if max_abs > tol:
        print(f'FAIL: max_abs_diff {max_abs:.4e} exceeds tol {tol:.4e}')
        return 1
    print('PASS: flash and torch attention paths agree within fp16 tolerance.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
