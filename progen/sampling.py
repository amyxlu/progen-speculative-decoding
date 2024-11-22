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


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    """ Original ProGen top-p sampling."""
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


def sample_vllm(device, model: LLM, tokenizer, context, max_length, num_return_sequences, top_p, temp):
    """Sample from the VLLM model."""
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_length,
    )
    if tokenizer is None:
        outputs = model.generate(context, sampling_params)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) == num_return_sequences
        output_texts = [output.text for output in outputs[0].outputs]
    else:
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        prompts = TokensPrompt(prompt_token_ids=input_ids)
        outputs = model.generate(prompts, sampling_params)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) == num_return_sequences
        tokens_batch = [output.token_ids for output in outputs[0].outputs]
        output_texts = tokenizer.decode_batch(tokens_batch)

    return output_texts, outputs


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


def compute_prompt_cross_entropy_vllm(llm: LLM, prompt: str, device, tokenizer=None) -> float:
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
