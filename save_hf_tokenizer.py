"""Saves a custom tokenizer in the HF format to be compatible with vLLM."""

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer


def convert_to_hf_tokenizer(custom_tokenizer):
    return PreTrainedTokenizerFast(tokenizer_object=custom_tokenizer)


def save_tokenizer_for_vllm(custom_tokenizer, save_path):
    hf_tokenizer = convert_to_hf_tokenizer(custom_tokenizer)
    hf_tokenizer.save_pretrained(save_path)


def create_tokenizer_custom(file):
    with open(file, "r") as f:
        return Tokenizer.from_str(f.read())


if __name__ == "__main__":
    # Load your custom tokenizer
    custom_tokenizer = create_tokenizer_custom(
        "/nas/ucb/ebronstein/progen-speculative-decoding/tokenizer.json"
    )

    # Save it in a format compatible with vLLM
    save_tokenizer_for_vllm(custom_tokenizer, "tokenizer")
