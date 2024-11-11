# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import torch

from vllm import LLM, SamplingParams


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    """ Original ProGen top-p sampling."""
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))

def sample_vllm(
        device, model: LLM, tokenizer, context, max_length, num_return_sequences, top_p,
        temp, pad_token_id):
    """Sample from the VLLM model."""
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_length,
    )
    hf_overrides = dict(
        pad_token_id=pad_token_id,
    )
    input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
    outputs = model.generate(input_ids, sampling_params, hf_overrides=hf_overrides)
    tokens_batch = [output.outputs[0].text for output in outputs]
    return tokenizer.decode_batch(tokens_batch)


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
