# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import argparse
import json
import pathlib

import torch

import benchmark_functions
from progen.sampling import compute_prompt_cross_entropy_vllm, sample, sample_vllm, cross_entropy, truncate
from progen.utils import create_model, create_tokenizer_custom, set_env, set_seed, print_time, get_benchmark_results_save_path


def none_or_val(value, dtype=str):
    return None if value == 'None' else dtype(value)


def main():

    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    speculative_models = models + ["[ngram]", None]

    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--fp16', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--context', type=str, default='1')
    parser.add_argument('--sanity', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--sample', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--benchmark', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_vllm', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--separate_tokenizer', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--speculative_model', type=none_or_val, choices=speculative_models, default=None)
    parser.add_argument('--num_speculative_tokens', type=lambda x: none_or_val(x, dtype=int), default=None)
    parser.add_argument('--ngram_prompt_lookup_max', type=int, default=4)
    parser.add_argument('--rope_dtype', type=str, default='float32')
    args = parser.parse_args()

    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'
    spec_model = args.speculative_model
    if spec_model is not None:
        if spec_model in models:
            spec_model = f'./checkpoints/{spec_model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    if args.separate_tokenizer or not args.use_vllm:
        with print_time('loading tokenizer'):
            tokenizer = create_tokenizer_custom(file='tokenizer.json')
    else:
        tokenizer = None

    with print_time('loading parameters'):
        model = create_model(
            ckpt=ckpt,
            fp16=args.fp16,
            use_vllm=args.use_vllm,
            tokenizer="tokenizer" if tokenizer is None else None,
            speculative_model=spec_model,
            num_speculative_tokens=args.num_speculative_tokens,
            ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
            rope_dtype=args.rope_dtype,
        )
        if not args.use_vllm:
            model = model.to(device)

    # (4) sanity

    if args.sanity:

        with print_time('sanity cross-entropy'):

            def ce(model, tokenizer, tokens, device):
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                        logits = model(target, labels=target).logits

                        # shift
                        logits = logits[:-1, ...]
                        target = target[1:]

                        return cross_entropy(logits=logits, target=target).item()

            x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
            x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
            x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'

            checkpoint_x_ce = {
                'progen2-small': (x_uniref90bfd30, 2.4),
                'progen2-medium': (x_uniref90bfd30, 1.9),
                'progen2-base': (x_uniref90bfd30, 1.9),
                'progen2-large': (x_uniref90bfd30, 1.8),
                'progen2-xlarge': (x_uniref90bfd30, 1.0),
                'progen2-oas': (x_oas, 0.3),
                'progen2-BFD90': (x_bfd90, 1.3),
            }

            prompt, ce_target = checkpoint_x_ce[args.model]

            if args.use_vllm:
                ce_eval = compute_prompt_cross_entropy_vllm(model, prompt, device=device, tokenizer=tokenizer)
            else:
                ce_eval = ce(model, tokenizer, prompt, device)

            print(ce_target, ce_eval, abs(ce_eval - ce_target))

            assert abs(ce_eval - ce_target) < 0.1

    # (5) sample

    if args.sample:
        with print_time('sampling'):
            if args.use_vllm:
                completions, outputs = sample_vllm(device=device, model=model, tokenizer=tokenizer, context=args.context, num_return_sequences=args.num_samples, temp=args.t, top_p=args.p, max_length=args.max_length)
            else:
                completions = sample(device=device, model=model, tokenizer=tokenizer, context=args.context, pad_token_id=tokenizer.encode('<|pad|>').ids[0], num_return_sequences=args.num_samples, temp=args.t, top_p=args.p, max_length=args.max_length)

            truncations = [truncate(completion, terminals=['1', '2']) for completion in completions]

            print(args.context)

            for (i, truncation) in enumerate(truncations):

                print()
                print(i)
                print(truncation)

    # (6) benchmark

    if args.benchmark:
        if args.use_vllm:
            assert tokenizer is None, 'Use --separate_tokenizer=False with --benchmark=True for vllm models'
            results = benchmark_functions.benchmark_vllm_model(
                model,
                tokenizer,
                args.context,
                device,
                args.max_length,
                args.num_samples,
                args.p,
                args.t,
            )
        else:
            raise NotImplementedError('benchmarking not implemented for non-VLLM models')

        # Add args to results
        results.update(vars(args))

        save_path = get_benchmark_results_save_path(
            root_dir='benchmark',
            model_name=args.model,
            use_vllm=args.use_vllm,
            num_samples=args.num_samples,
            max_len=args.max_length,
            speculative_model=args.speculative_model,
        )
        if not pathlib.Path(save_path).parent.exists():
            pathlib.Path(save_path).parent.mkdir(parents=True)
        with open(save_path, 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()
    print('done.')
