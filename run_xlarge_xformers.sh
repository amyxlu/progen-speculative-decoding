VLLM_ATTENTION_BACKEND=XFORMERS python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=True --sanity=False --benchmark true --bsn xformers_fp16_r32 --fp16 true --rope_dtype float32
VLLM_ATTENTION_BACKEND=XFORMERS python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=True --sanity=False --benchmark true --bsn xformers_fp16_r16 --fp16 true --rope_dtype float16
VLLM_ATTENTION_BACKEND=XFORMERS python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=True --sanity=False --benchmark true --bsn xformers_fp32_r32 --fp16 False --rope_dtype float32
VLLM_ATTENTION_BACKEND=XFORMERS python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=True --sanity=False --benchmark true --bsn xformers_fp32_r16 --fp16 False --rope_dtype float16