#!/bin/bash


### Speculative decoding with xlarge target model for different NUM_SPEC_TOKENS and MAX_LENGTH. NUM_SAMPLES=1.

for ((ntoks=1; ntoks<10; ntoks+=1)); do
    MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=$ntoks NUM_SAMPLES=32 MAX_LENGTH=64 BENCHMARK=False LOG_TO_WANDB=True SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh
done

# for ((ntoks=1; ntoks<10; ntoks+=1)); do
#     CUDA_VISIBLE_DEVICES=2 MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=$ntoks NUM_SAMPLES=16 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh
# done
