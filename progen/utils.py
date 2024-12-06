import random
import time
import os
import abc
import datetime
from typing import Tuple, Union

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers.cache_utils import DynamicCache
from torch.nn import functional as F
from vllm import LLM

from .modeling_progen import ProGenForCausalLM
from .modeling_progen_vllm import ProGenForCausalLM as ProGenForCausalLMVLLM


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_benchmark_results_save_dir(
    root_dir, model_name, use_vllm, num_samples, max_len, speculative_model,
    add_timestamp=True, bsn=None,
):
    path = f"{model_name}/vllm_{use_vllm}_samples_{num_samples}_len_{max_len}"
    if speculative_model is not None:
        path += f"_spec_{speculative_model}"

    if bsn is not None:
        path += f"_{bsn}"
    elif add_timestamp:
        path += f"-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return os.path.join(root_dir, path)


########################################################################
# model


def create_model(
    ckpt,
    fp16=True,
    use_vllm=False,
    tokenizer=None,
    speculative_model=None,
    num_speculative_tokens=None,
    ngram_prompt_lookup_min=1,
    ngram_prompt_lookup_max=4,
    rope_dtype="float32",
    disable_log_stats=False,
    use_cache=False,
):
    if use_vllm:
        assert (speculative_model is None) == (num_speculative_tokens is None), (
            "speculative_model and num_speculative_tokens must be both None or both "
            "not None"
        )
        if speculative_model is not None:
            assert (
                num_speculative_tokens > 0
            ), "num_speculative_tokens must be greater than 0"

        from vllm import ModelRegistry
        ModelRegistry.register_model("ProGenForCausalLM", ProGenForCausalLMVLLM)
        hf_overrides = {"rope_dtype": rope_dtype}
        return LLM(
            model=ckpt,
            tokenizer=tokenizer,
            dtype="float16" if fp16 else "auto",
            skip_tokenizer_init=tokenizer is None,
            trust_remote_code=False,
            hf_overrides=hf_overrides,
            speculative_model=speculative_model,
            num_speculative_tokens=num_speculative_tokens,
            ngram_prompt_lookup_min=ngram_prompt_lookup_min,
            ngram_prompt_lookup_max=ngram_prompt_lookup_max,
            disable_log_stats=disable_log_stats,
        )

    assert rope_dtype == "float32", "rope_dtype must be float32 when not using VLLM"
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True, use_cache=use_cache)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt, use_cache=use_cache)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass


class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def __init__(self, temperature: float = 1):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)


class MultinomialProcessor(LogitsProcessor):
    """Multinomial: Random sampling."""

    def __init__(self, temperature: float):
        super().__init__(temperature)

    def _process(self, logits: Tensor) -> Tensor:
        return logits

    def sample(self, probs: Tensor) -> Tensor:
        return torch.multinomial(probs, num_samples=1)


class TopKProcessor(MultinomialProcessor):
    """Top-k: Top-k sampling."""

    def __init__(self, temperature: float, top_k: int):
        super().__init__(temperature)
        self.top_k = top_k

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        return logits


class NucleusProcessor(MultinomialProcessor):
    """Nucleus: Top-p sampling."""

    def __init__(self, temperature: float, top_p: float):
        super().__init__(temperature)
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


####
# https://github.com/romsto/Speculative-Decoding/blob/main/utils/logits_processor.py
###

class TopKNucleusProcessor(MultinomialProcessor):
    """Top-k and nucleus: Top-k sampling with top-p fallback."""

    def __init__(self, temperature: float, top_k: int, top_p: float):
        super().__init__(temperature)
        self.top_k = top_k
        self.top_p = top_p

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e20
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits


def prune_cache(cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end.

    Args:
        cache (Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]: The pruned KV cache.
    """
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError("Unsupported cache type.")


def prune_tuple_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for most models.
    It works for models having past_key_values such as Tuple of tuple(Tensor) of length n_layers, containing 2 or 4 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)

    Args:
        cache Tuple(Tuple[Tensor, Tensor]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Tuple[Tensor, Tensor]: The pruned KV cache.
    """
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)


def prune_dynamic_cache(cache: DynamicCache, num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for models using DynamicCache.

    Args:
        cache (DynamicCache): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        DynamicCache: The pruned KV cache. (same instance as the input cache, but modified in place)
    """
    if cache is None:
        return None

    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
    cache._seen_tokens -= num_tokens_to_discard

    return cache
