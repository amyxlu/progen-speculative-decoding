#!/bin/bash

MODEL=${MODEL:-"progen2-small"}
NUM_SAMPLES=${NUM_SAMPLES:-1}
MAX_LENGTH=${MAX_LENGTH:-8}

# Sampling
TOP_P=${TOP_P:-0.95}
TEMP=${TEMP:-0.2}
FREQ_PENALTY=${FREQ_PENALTY:-0.0}

# Speculative decoding
SPEC_MODEL=${SPEC_MODEL:-"None"}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:="None"}
NGRAM_PROMPT_LOOKUP_MAX=${NGRAM_PROMPT_LOOKUP_MAX:-4}

USE_VLLM=${USE_VLLM:-True}
SEPARATE_TOKENIZER=${SEPARATE_TOKENIZER:-False}

# Run mode
SANITY=${SANITY:-False}
SAMPLE=${SAMPLE:-False}
BENCHMARK=${BENCHMARK:-True}
LOG_SPEC_DECODE_METRICS=${LOG_SPEC_DECODE_METRICS:-False}
LOG_TO_WANDB=${LOG_TO_WANDB:-False}

python sample.py \
    --model=$MODEL \
    --num-samples=$NUM_SAMPLES \
    --max-length=$MAX_LENGTH \
    --p=$TOP_P \
    --t=$TEMP \
    --frequency_penalty=$FREQ_PENALTY \
    --speculative_model=$SPEC_MODEL \
    --num_speculative_tokens=$NUM_SPEC_TOKENS \
    --ngram_prompt_lookup_max=$NGRAM_PROMPT_LOOKUP_MAX \
    --use_vllm=$USE_VLLM \
    --separate_tokenizer=$SEPARATE_TOKENIZER \
    --sanity=$SANITY \
    --sample=$SAMPLE \
    --benchmark=$BENCHMARK \
    --log_spec_decode_metrics=$LOG_SPEC_DECODE_METRICS \
    --log_to_wandb=$LOG_TO_WANDB
