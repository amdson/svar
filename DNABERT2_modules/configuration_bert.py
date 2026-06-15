# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from transformers.configuration_utils import PretrainedConfig


class BertConfig(PretrainedConfig):

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
        attn_impl: str = 'flash',
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Only used by the `triton`/`torch` attention paths (the `flash` path computes ALiBi on the fly).
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
            attn_impl (str): Which self-attention implementation to use. One of:
                - 'flash': FlashAttention >= 2.4 varlen kernel with native ALiBi slopes (default). Requires
                    CUDA + fp16/bf16 and the `flash-attn` package. No dense bias is materialized.
                - 'triton': the legacy Triton kernel that takes a dense ALiBi+mask bias. Requires CUDA +
                    fp16/bf16 and dropout == 0.
                - 'torch': pure-PyTorch matmul attention with a dense bias. Works on CPU/fp32; used as a
                    fallback for tests and debugging.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size
        self.attn_impl = attn_impl
