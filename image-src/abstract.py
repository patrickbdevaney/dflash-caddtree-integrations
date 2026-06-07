# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import abstractmethod
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import get_mamba_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


def _num_speculative_blocks(vllm_config: VllmConfig) -> int:
    """Per-sequence extra GDN state-block reservation (INVARIANT 1).

    Linear DFlash reserves num_speculative_tokens blocks. DDTree tree
    verification needs one recurrent-state slot per tree node, so reserve the
    tree node budget when it exceeds num_spec. Tree width/budget is taken from
    env during development (DFLASH_NODE_BUDGET); when unset this is a strict
    no-op (== num_speculative_tokens) so linear behavior is byte-identical.
    """
    if vllm_config.speculative_config is None:
        return 0
    num_spec = vllm_config.speculative_config.num_speculative_tokens or 0
    try:
        node_budget = int(os.environ.get("DFLASH_NODE_BUDGET", "0"))
    except ValueError:
        node_budget = 0
    return max(num_spec, node_budget)


class MambaBase(AttentionLayerBase):
    """
    Base class for Mamba-like layers which support the v1 engine.
    Inherit from this class if you implement a custom layer.
    """

    # Contains the KV cache (mamba state) for the layer
    # in the shape specified by `self.get_state_shape`.
    kv_cache: tuple[torch.Tensor, ...]

    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """
        Defines the shape of the state.
        For mamba layers this is usually a (conv_state, ssm_state) tuple.
        In this case, returns (conv_state_shape, ssm_state_shape).
        """
        pass

    @property
    @abstractmethod
    def mamba_type(self) -> str:
        pass

    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        pass

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        assert mamba_block_size is not None
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
            num_speculative_blocks=_num_speculative_blocks(vllm_config),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this Mamba layer."""
        return get_mamba_attn_backend(self.mamba_type)
