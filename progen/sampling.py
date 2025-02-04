# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import torch

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from progen import speculative
from progen import utils


def prepare_input_for_vllm_model(input, tokenizer, device):
    """Prepare input for a VLLM model."""
    # No need to tokenize if tokenizer is None because the tokenization is done in the
    # model.
    if tokenizer is None:
        return input

    input_ids = torch.tensor(tokenizer.encode(input).ids).view([1, -1]).to(device)
    return TokensPrompt(prompt_token_ids=input_ids)


def prepare_input_for_non_vllm_model(input, tokenizer, device):
    """Prepare input for a non-VLLM model."""
    return torch.tensor(tokenizer.encode(input).ids).view([1, -1]).to(device)


def prepare_input_for_model(input, model, tokenizer, device):
    """Prepare input for the model."""
    if isinstance(model, LLM):
        return prepare_input_for_vllm_model(input, tokenizer, device)
    else:
        return prepare_input_for_non_vllm_model(input, tokenizer, device)


def process_outputs_from_vllm_model(outputs, tokenizer):
    """Process output from a VLLM model."""
    assert len(outputs) == 1
    if tokenizer is None:
        return [output.text for output in outputs[0].outputs]
    else:
        tokens_batch = [output.token_ids for output in outputs[0].outputs]
        return tokenizer.decode_batch(tokens_batch)


def process_outputs_from_non_vllm_model(outputs, tokenizer):
    """Process output from a non-VLLM model."""
    # outputs is a tensor of shape [num_return_sequences, max_length] if speculative
    # decoding is not used. Otherwise, outputs is a list of length 1, which contains
    # the token IDs as integers.
    if isinstance(outputs, torch.Tensor):
        outputs = [
            outputs[i, ...].detach().cpu().numpy().tolist()
            for i in range(outputs.shape[0])
        ]
    return tokenizer.decode_batch(outputs)


def process_outputs_from_model(outputs, model, tokenizer):
    """Process output from the model."""
    if isinstance(model, LLM):
        return process_outputs_from_vllm_model(outputs, tokenizer)
    else:
        return process_outputs_from_non_vllm_model(outputs, tokenizer)


def make_generate_fn(
    model,
    max_length,
    num_return_sequences,
    top_p,
    temp,
    pad_token_id=None,
    eos_token_id=None,
    spec_model=None,
    num_speculative_tokens=None,
    frequency_penalty=None,
    logits_processor_type: utils.LogitsProcessorType = "greedy",
):
    """Make a generate function for the model."""
    if isinstance(model, LLM):
        if spec_model is None:
            # VLLM model without speculative decoding or with VLLM-based speculative decoding.
            sampling_params = SamplingParams(
                n=num_return_sequences,
                temperature=temp,
                top_p=top_p,
                max_tokens=max_length,
                frequency_penalty=frequency_penalty,
            )

            def generate(input_ids):
                return model.generate(input_ids, sampling_params)
        else:
            # VLLM model with custom (non-VLLM-based) speculative decoding.
            # TODO: Implement this. Custom speculative decoding does not support VLLM models yet.
            raise NotImplementedError(
                "Custom speculative decoding does not support VLLM models yet."
            )

            def generate(input_ids):
                tokens, acceptance_rate = speculative.speculative_generate(
                    inputs=input_ids,
                    drafter=spec_model,
                    target=model,
                    gamma=num_speculative_tokens,
                    logits_processor=logits_processor,
                    max_gen_len=max_length,
                    eos_tokens_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                return [tokens]

    else:
        assert (
            pad_token_id is not None
        ), "pad_token_id must be provided for non-VLLM models"
        assert (
            frequency_penalty is None
        ), "Frequency penalty is only supported for VLLM models"

        if spec_model is None:
            # Use greedy decoding if temp == 0 (do_sample=False).
            if temp == 0:
                do_sample = False
                temp = None
                top_p = None
            else:
                do_sample = True

            def generate(input_ids):
                return model.generate(
                    input_ids,
                    do_sample=do_sample,
                    temperature=temp,
                    max_length=max_length,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=pad_token_id,
                )

        else:
            assert (
                num_return_sequences == 1
            ), "Speculative decoding without vllm only supports num_return_sequences=1"
            assert (
                eos_token_id is not None
            ), "eos_token_id must be provided when using speculative decoding"
            assert (
                num_speculative_tokens is not None
            ), "num_speculative_tokens must be provided when using speculative decoding"

            logits_processor = utils.make_logits_processor(
                logits_processor_type, temperature=temp, top_p=top_p
            )

            def generate(input_ids):
                tokens, acceptance_rate = speculative.speculative_generate(
                    inputs=input_ids,
                    drafter=spec_model,
                    target=model,
                    gamma=num_speculative_tokens,
                    logits_processor=logits_processor,
                    max_gen_len=max_length,
                    eos_tokens_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                return [tokens]

    return generate


def sample(
    device,
    model,
    tokenizer,
    context,
    max_length,
    num_return_sequences,
    top_p,
    temp,
    pad_token_id=None,
    eos_token_id=None,
    spec_model=None,
    num_speculative_tokens=None,
    frequency_penalty=None,
    logits_processor_type="greedy",
):
    """Original ProGen top-p sampling."""
    # if spec_model is None:
    #     def generate(input_ids):
    #         return model.generate(
    #             input_ids,
    #             do_sample=True,
    #             temperature=temp,
    #             max_length=max_length,
    #             top_p=top_p,
    #             num_return_sequences=num_return_sequences,
    #             pad_token_id=pad_token_id,
    #         )
    # else:
    #     assert num_return_sequences == 1, "Speculative decoding without vllm only supports num_return_sequences=1"
    #     def generate(input_ids):
    #         tokens, acceptance_rate = speculative.speculative_generate(
    #             inputs=input_ids,
    #             drafter=spec_model,
    #             target=model,
    #             gamma=num_speculative_tokens,
    #             max_gen_len=max_length,
    #             eos_tokens_id=eos_token_id,
    #             pad_token_id=pad_token_id,
    #         )
    #         return [tokens]
    generate = make_generate_fn(
        model,
        max_length,
        num_return_sequences,
        top_p,
        temp,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        spec_model=spec_model,
        num_speculative_tokens=num_speculative_tokens,
        frequency_penalty=frequency_penalty,
        logits_processor_type=logits_processor_type,
    )

    with torch.no_grad():
        # [1, 1]
        # input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        input_ids = prepare_input_for_model(context, model, tokenizer, device)

        # [num_samples, max_length]
        tokens_batch = generate(input_ids)

        # if isinstance(tokens_batch, torch.Tensor):
        #     tokens_batch = [
        #         tokens_batch[i, ...].detach().cpu().numpy().tolist()
        #         for i in range(tokens_batch.shape[0])
        #     ]
        # return tokenizer.decode_batch(tokens_batch)
        return process_outputs_from_model(tokens_batch, model, tokenizer)


# TODO: probably delete this function.
def sample_vllm(
    device,
    model: LLM,
    tokenizer,
    context,
    max_length,
    num_return_sequences,
    top_p,
    temp,
    frequency_penalty,
):
    """Sample from the VLLM model."""
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_length,
        frequency_penalty=frequency_penalty,
    )
    input = prepare_input_for_model(context, model, tokenizer, device)
    outputs = model.generate(input, sampling_params)
    output_texts = process_outputs_from_model(outputs, model, tokenizer)

    # if tokenizer is None:
    #     outputs = model.generate(context, sampling_params)
    #     assert len(outputs) == 1
    #     assert len(outputs[0].outputs) == num_return_sequences
    #     output_texts = [output.text for output in outputs[0].outputs]
    # else:
    #     input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
    #     prompts = TokensPrompt(prompt_token_ids=input_ids)
    #     outputs = model.generate(prompts, sampling_params)
    #     assert len(outputs) == 1
    #     assert len(outputs[0].outputs) == num_return_sequences
    #     tokens_batch = [output.token_ids for output in outputs[0].outputs]
    #     output_texts = tokenizer.decode_batch(tokens_batch)

    assert (
        len(output_texts) == num_return_sequences
    ), f"Expected {num_return_sequences} outputs, got {len(output_texts)}"
    return output_texts, outputs


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[: (min(pos) + 1)]
    else:
        return sample


def cross_entropy(logits, target, reduction="mean"):
    return torch.nn.functional.cross_entropy(
        input=logits,
        target=target,
        weight=None,
        size_average=None,
        reduce=None,
        reduction=reduction,
    )


def compute_prompt_cross_entropy_vllm(
    llm: LLM, prompt: str, device, tokenizer=None
) -> float:
    """Computes the cross-entropy of a prompt with the model.

    The prompt should already be prepended with either a 1 or 2 token.
    """
    # Set prompt_logprobs=0 to only compute the logprobs of the prompt tokens.
    # Set max_tokens=1 to only generate one token for speed.
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)

    if tokenizer is not None:
        input_ids = torch.tensor(tokenizer.encode(prompt).ids).to(device)
        tokens_prompt = TokensPrompt(prompt_token_ids=input_ids)
    else:
        tokens_prompt = prompt

    output = llm.generate(tokens_prompt, sampling_params)
    # There should only be one output sequence.
    assert len(output) == 1
    # The prompt logprobs should be the same length as the prompt.
    assert len(output[0].prompt_logprobs) == len(prompt)

    prompt_logprobs = []
    # Skip the first logprob, which is None because it corresponds to the first token.
    # Each subsequent logprob is a dict of length 1 where the key is the token ID and
    # the value is a Logprob object.
    for i, token_id_to_logprob in enumerate(output[0].prompt_logprobs[1:], start=1):
        assert len(token_id_to_logprob) == 1
        logprob = list(token_id_to_logprob.values())[0].logprob
        prompt_logprobs.append(logprob)

    return -np.mean(prompt_logprobs)
