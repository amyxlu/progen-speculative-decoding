#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --output=jobs/ragged-%j.out


MODEL=${MODEL:-"progen2-xlarge"}
NUM_SAMPLES=${NUM_SAMPLES:-1}
MAX_LENGTH=${MAX_LENGTH:-8}

# Sampling
TOP_P=${TOP_P:-0.95}
TEMP=${TEMP:-0.2}
FREQ_PENALTY=${FREQ_PENALTY:-0.0}

# Speculative decoding
SPEC_MODEL=${SPEC_MODEL:-"progen2-small"}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:="None"}
NGRAM_PROMPT_LOOKUP_MAX=${NGRAM_PROMPT_LOOKUP_MAX:-4}

USE_VLLM=${USE_VLLM:-False}
SEPARATE_TOKENIZER=${SEPARATE_TOKENIZER:-False}

# Run mode
SANITY=${SANITY:-False}
SAMPLE=${SAMPLE:-False}
BENCHMARK=${BENCHMARK:-True}
LOG_SPEC_DECODE_METRICS=${LOG_SPEC_DECODE_METRICS:-False}

BATCH_SIZE=${BATCH_SIZE:-1}

cd ~/progen-speculative-decoding
source .venv/bin/activate

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
    --batch_size=$BATCH_SIZE \
    --fp16 False \
    --ragged-batches true
