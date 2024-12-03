import time

import torch



# Adapted from:
# https://github.com/triton-lang/triton/blob/57643b3f4746b3f53334fd6ce8020dd6c902c7f4/python/triton/testing.py#L95
def custom_do_bench(
    fn,
    n_warmup=3,
    warmup_time=None,
    n_repeat=10,
    rep_time=None,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
    device_type="cuda",
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """
    assert (warmup_time is None) != (
        n_warmup is None
    ), "Either warmup xor warmup_time should be provided"
    assert (rep_time is None) != (
        n_repeat is None
    ), "Either rep xor rep_time should be provided"
    assert return_mode in ["min", "max", "mean", "median"]

    # di = torch._dynamo.device_interface.get_interface_for_device(device_type)
    # The above line requires torch 2.0 (torch dynamo) but I devlieve we can replace it with the following for torch 1.9
    di = torch.cuda

    fn()
    di.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device_type)

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    if n_warmup is None:
        n_warmup = max(1, int(warmup_time / estimate_ms))
    if n_repeat is None:
        n_repeat = max(1, int(rep_time / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times_list = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times = torch.tensor(times_list, dtype=torch.float)
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(quantile_values) == 1:
            quantile_values = quantile_values[0]
    else:
        quantile_values = None
    return torch.mean(times).item(), torch.std(times).item(), quantile_values, times_list


def flop_counter(model, input_data):
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model, input_data)
    return flops


def benchmark_vllm_model(
    model,
    tokenizer,
    context,
    device,
    max_length,
    num_samples,
    top_p,
    temp,
    frequency_penalty,
    n_warmup=3,
    n_repeat=10,
):
    from vllm import SamplingParams
    from vllm.inputs.data import TokensPrompt
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temp,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        max_tokens=max_length,
        detokenize=False,  # Do not detokenize for benchmarking
    )
    if tokenizer is None:
        prompts = context
    else:
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        prompts = TokensPrompt(prompt_token_ids=input_ids)

    def generate():
        model.generate(prompts, sampling_params)

    # Input kwargs
    kwargs = dict(
        n_warmup=n_warmup,
        warmup_time=None,
        n_repeat=n_repeat,
        rep_time=None,
        grad_to_none=None,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        fast_flush=True,
        return_mode="mean",
        device_type="cuda",
    )

    avg_time, std_dev, quantiles, times = custom_do_bench(generate, **kwargs)
    print(f"Average time: {avg_time} ms, Standard deviation: {std_dev} ms")
    print(f"Quantiles: {quantiles}")
    print(f"Times: {times}")

    return {
        "avg_time": avg_time,
        "std": std_dev,
        "quantiles": quantiles,
        "times": times,
        "do_bench_kwargs": kwargs,
    }


def benchmark_batch_spec_model(
    draft_model,
    target_model,
    tokenizer,
    context,
    device,
    max_length,
    num_samples,
    top_p,
    temp,
    frequency_penalty,
    batch_size,
    num_speculative_tokens,
    n_warmup=3,
    n_repeat=10,
):
    from progen.speculative import speculative_generate_batched
    from progen.utils import NucleusProcessor
    assert frequency_penalty == 0, "Frequency penalty is not supported for batched speculative decoding."
    pad_token_id=tokenizer.encode('<|pad|>').ids[0] 
    def generate():
        for i in range(0, num_samples, batch_size):
            input_ids = [tokenizer.encode(context).ids.copy() for _ in range(min(batch_size, num_samples - i))]
            ids_list, accept_rate = speculative_generate_batched(
                inputs = input_ids,
                drafter = draft_model,
                target = target_model,
                tokenizer = tokenizer,
                gamma = num_speculative_tokens,
                logits_processor = NucleusProcessor(temp, top_p),
                max_gen_len = max_length,
                eos_tokens_id = [],
                pad_token_id = pad_token_id,
                use_cache = True,
                skip_sample_adjustment = False,
                first_target = True, 
                debug = False,
            )

    # Input kwargs
    kwargs = dict(
        n_warmup=n_warmup,
        warmup_time=None,
        n_repeat=n_repeat,
        rep_time=None,
        grad_to_none=None,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        fast_flush=True,
        return_mode="mean",
        device_type="cuda",
    )

    avg_time, std_dev, quantiles, times = custom_do_bench(generate, **kwargs)
    print(f"Average time: {avg_time} ms, Standard deviation: {std_dev} ms")
    print(f"Quantiles: {quantiles}")
    print(f"Times: {times}")

    return {
        "avg_time": avg_time,
        "std": std_dev,
        "quantiles": quantiles,
        "times": times,
        "do_bench_kwargs": kwargs,
    }


def collect_speculative_decoding_metrics(
    model,
    tokenizer,
    context,
    device,
    max_length,
    num_samples,
    top_p,
    temp,
    frequency_penalty,
    n_repeat=10,
):
    from vllm import SamplingParams
    from vllm.inputs.data import TokensPrompt
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temp,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        max_tokens=max_length,
        detokenize=False,  # Do not detokenize for benchmarking
    )
    if tokenizer is None:
        prompts = context
    else:
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        prompts = TokensPrompt(prompt_token_ids=input_ids)

    for _ in range(n_repeat):
        model.generate(prompts, sampling_params)

    # Sleep for 5 seconds to allow the metrics to be logged. 5 seconds is vllm's default
    # logging interval.
    time.sleep(5)


def main():
    # Initialize your model and input data. Put them on the correct device!
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
    ).cuda()
    input_data = torch.randn((64, 32)).cuda()  # Example input shape

    # Wrap the forward pass in a function
    def forward_pass():
        with torch.no_grad():  # Ensure no gradients are calculated
            output = model(input_data)
        return output

    # Benchmark the forward pass
    # avg_time, std_dev = triton.testing.do_bench(forward_pass)
    avg_time, std_dev = custom_do_bench(forward_pass)
    print(f"Average time: {avg_time} ms, Standard deviation: {std_dev} ms")

    flops = flop_counter(model, input_data)
    print(f"FLOPs: {flops.total()}")
    print("By module and operator:", flops.by_module_and_operator())


if __name__ == "__main__":
    main()
