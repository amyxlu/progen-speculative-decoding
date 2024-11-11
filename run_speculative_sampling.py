# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import argparse
import time
import torch

from progen.sampling import sample, cross_entropy, truncate
from progen.speculative import speculative_generate
from progen.utils import create_model, create_tokenizer_custom, set_env, set_seed, print_time, GreedyProcessor, LogitsProcessor
import progen.printing_utils as printing
from termcolor import colored


def main():

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--draft_model', type=str, choices=models_151M + models_754M, default='progen2-small')
    parser.add_argument('--target_model', type=str, choices=models_2B + models_6B, default='progen2-xlarge')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, nargs="+", default='1', help="Defaults to Progen's BOS token.")
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    # progen special tokens:
    #  '<|pad|>': 0,
    #  '<|bos|>': 1,
    #  '<|eos|>': 2,

    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    draft_ckpt = f'./checkpoints/{args.draft_model}'
    target_ckpt = f'./checkpoints/{args.target_model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading parameters'):
        draft_model = create_model(ckpt=draft_ckpt, fp16=args.fp16).to(device)
        target_model = create_model(ckpt=target_ckpt, fp16=args.fp16).to(device)

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json')

    inputs = args.context
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.tensor([int(i) for i in inputs], dtype=torch.long, device=device)

    # (4) speculative sampling
    eos_tok = tokenizer.token_to_id("<|eos|>")
    bos_tok = tokenizer.token_to_id("<|bos|>")
    pad_tok = tokenizer.token_to_id("<|pad|>")

    spec_start_time = time.time()
    ids, accept_rate = speculative_generate(
        inputs = inputs,
        drafter = draft_model,
        target = target_model,
        tokenizer = tokenizer,
        gamma = 5,
        logits_processor = GreedyProcessor(),
        max_gen_len = args.max_length,
        eos_tokens_id = eos_tok,
        pad_token_id = pad_tok,
        use_cache = False,
        skip_sample_adjustment = False,
        first_target = True, 
        debug = False,
    )
    
    spec_end_time = time.time()
    spec_output = tokenizer.decode(ids, skip_special_tokens=True)

    print(colored("========== Speculative ==========", "green"))
    print(colored("Out:", "green"), spec_output)
    print(colored(f"Acceptance rate: {accept_rate:.3f}", "green"))

    spec_throughput = len(spec_output) / (spec_end_time - spec_start_time)
    print(colored(f"Time: {spec_end_time - spec_start_time:.1f}s", "green"))
    print(colored(f"Throughput: {spec_throughput:.1f} tokens/s", "green"))
    print(colored("========== Speculative ==========", "green")) 
     

if __name__ == '__main__':
    main()
    print('done.')
