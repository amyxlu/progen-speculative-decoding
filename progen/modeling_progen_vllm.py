import torch
import torch.nn as nn
from typing import Iterable, List, Optional, Tuple, Union

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from progen.configuration_progen import ProGenConfig


ROPE_THETA = 10000


# This function is implement on the vllm main branch but not in the released version.
def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


class ProGenAttention(nn.Module):
    def __init__(
        self,
        config: ProGenConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.out_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=config.rotary_dim,
            max_position=config.max_position_embeddings,
            base=ROPE_THETA,
            # TODO: are these two args correct?
            is_neox_style=True,
            rope_scaling=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class ProGenMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: ProGenConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class ProGenBlock(nn.Module):
    def __init__(
        self,
        config: ProGenConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = ProGenAttention(config)
        self.mlp = ProGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            positions,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        return hidden_states


class ProGenModel(nn.Module):
    def __init__(
        self,
        config: ProGenConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: ProGenBlock(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.n_embd
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = self.wte(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.h[i]
            hidden_states = layer(
                hidden_states, positions, kv_caches[i - self.start_layer], attn_metadata
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class ProGenForCausalLM(nn.Module):
    def __init__(
        self,
        config: ProGenConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.transformer = ProGenModel(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "transformer"),
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=True)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask.
                # NOTE: "c_attn.bias" should not be skipped.
                continue
            if not name.startswith("transformer.") and not name.startswith("lm_head."):
                name = "transformer." + name

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
