import torch
import torch.nn as nn
from typing import Iterable, List, Optional, Tuple, Union

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
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
        config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hf_config = config.hf_text_config
        self.hidden_size = config.get_hidden_size()
        # TODO: maybe replace with `config.get_num_attention_heads()` and pass in the
        # parallel_config.
        total_num_heads = hf_config.num_attention_heads
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

        # ProGen2 uses GPT-J-style rotary embeddings, not Neox-style, so set
        # is_neox_style=False.
        rope_dtype = hf_config.rope_dtype
        if rope_dtype == "float32":
            rope_dtype = torch.float32
        elif rope_dtype == "float16":
            rope_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported rope_dtype: {rope_dtype}. Must be float32 or float16.")
        # ProGen2 computes RoPE in fp32, so set dtype=torch.float32.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=hf_config.rotary_dim,
            max_position=hf_config.max_position_embeddings,
            base=ROPE_THETA,
            is_neox_style=False,
            rope_scaling=None,
            dtype=rope_dtype,
        )

    def _split_heads(
        self, x: torch.Tensor, num_heads: int, head_dim: int, mp_num: int
    ) -> torch.Tensor:
        # This function is copied from the original ProGen2 model code.
        # x: [(batch_size), seq_len, mp_num, hidden_size / mp_num]
        # reshaped: [(batch_size), seq_len, mp_num, num_heads / mp_num, head_dim]
        reshaped = x.reshape(x.shape[:-1] + (num_heads // mp_num, head_dim))
        # reshaped: [(batch_size), seq_len, num_heads, head_dim]
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(
            hidden_states
        )  # [(batch_size), seq_len, 3 * hidden_size]
        # TODO: remove commented out torch.save calls after debugging is complete.
        # torch.save(qkv, "qkv.pt")

        # TODO: is this more complex splitting and reshaping (from original model code)
        # necessary? Can it be simplified?
        # NOTE: this complex splitting and reshaping code is copied from the original
        # ProGen2 model code to ensure compatibility. Simply doing
        # `q, k, v = torch.chunk(qkv, 3, dim=-1)` does not work.
        mp_num = 8
        # [(batch_size), seq_len, mp_num, 3 * hidden_size / mp_num]
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))
        # torch.save(qkv_split, "qkv_split.pt")

        # NOTE: despite the name, qkv stores the tensors in the order q, v, k.
        # q, v, k are [(batch_size), seq_len, mp_num, hidden_size / mp_num]
        q, v, k = torch.chunk(qkv_split, 3, dim=-1)

        # [(batch_size), seq_len, num_heads, head_dim]
        q = self._split_heads(q, self.num_heads, self.head_dim, mp_num=mp_num)
        k = self._split_heads(k, self.num_heads, self.head_dim, mp_num=mp_num)
        v = self._split_heads(v, self.num_heads, self.head_dim, mp_num=mp_num)

        # torch.save(q, "q_pre_rope.pt")
        # torch.save(k, "k_pre_rope.pt")
        # torch.save(v, "v_pre_rope.pt")

        # TODO: can we do `q, k = self.rotary_emb(positions, q, k)` instead?
        # It currently raises an error:
        # RuntimeError: Error in model execution: CUDA error: an illegal memory access was encountered
        # Cast q and k to float32 before passing to rotary_emb to make the computation
        # happen in fp32 to match the original ProGen2 model. Without this cast,
        # rotary_emb will convert the sin/cos positional embeddings to the dtype of q
        # and k (fp16 by default).
        q, k = self.rotary_emb.forward_native(
            positions, q.to(self.rotary_emb.dtype), k.to(self.rotary_emb.dtype)
        )
        # torch.save(q, "q_post_rope.pt")
        # torch.save(k, "k_post_rope.pt")

        # TODO: is this reshape needed?
        # Cast q, k, v back to the original dtype (fp16 by default) before passing to
        # the attention layer because FlashAttion (default attention backend) expects
        # the input tensors to be in fp16 or bfloat16.
        q = q.reshape(q.shape[:-2] + (-1,)).to(qkv.dtype)
        v = v.reshape(v.shape[:-2] + (-1,)).to(qkv.dtype)
        k = k.reshape(k.shape[:-2] + (-1,)).to(qkv.dtype)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        # torch.save(attn_output, "attn_output.pt")

        output, _ = self.out_proj(attn_output)
        # torch.save(output, "attn_output_post_proj.pt")
        return output


class ProGenMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: ModelConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hf_config = config.hf_text_config
        hidden_size = config.get_hidden_size()
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
        self.act = get_act_fn(hf_config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class ProGenBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hf_config = config.hf_text_config
        hidden_size = config.get_hidden_size()
        inner_dim = hf_config.n_inner if hf_config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=hf_config.layer_norm_epsilon)
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
        config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hf_config = config.hf_text_config
        self.config = config
        vocab_size = config.get_vocab_size()
        self.vocab_size = vocab_size
        self.embed_dim = config.get_hidden_size()

        self.wte = VocabParallelEmbedding(vocab_size, self.embed_dim)
        # TODO: should this use `config.get_num_layers()` instead of
        # `hf_config.num_hidden_layers`?
        self.start_layer, self.end_layer, self.h = make_layers(
            hf_config.num_hidden_layers,
            lambda prefix: ProGenBlock(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=hf_config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.embed_dim
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
        vllm_config: VllmConfig = None,
        prefix: str = "",
    ):
        super().__init__()
        model_config = vllm_config.model_config
        self.config = model_config
        self.transformer = ProGenModel(
            model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "transformer"),
        )
        vocab_size = model_config.get_vocab_size()
        self.lm_head = ParallelLMHead(
            vocab_size, model_config.get_hidden_size(), bias=True
        )
        self.logits_processor = LogitsProcessor(vocab_size)
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
