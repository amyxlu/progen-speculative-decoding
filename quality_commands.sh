#!/bin/bash

# This script contains the commands to reproduce all of Eli's baseline and speculative
# decoding experiments.
# NUM_SAMPLES: number of sequences/samples to generate.
# MAX_LENGTH: maximum length of each sample.
# MODEL: target model to use for generation.
# SPEC_MODEL: draft model to use for speculative decoding.
# NUM_SPEC_TOKENS: number of speculative tokens to generate (draft sequence length).
# TEMP: temperature for sampling.
# TOP_P: top-p sampling threshold.
# FREQ_PENALTY: frequency penalty for speculative decoding.

# Baseline: progen2-small, medium, and xlarge for different NUM_SAMPLES and MAX_LENGTH.

# spec_decoding_xlarge - 0
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 1
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 2
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 3
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 4
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 5
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 6
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 7
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 8
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 9
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 10
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 11
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 12
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 13
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 14
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 15
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 16
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 17
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 18
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 19
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 20
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 21
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 22
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 23
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 24
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 25
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 26
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 27
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 28
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 29
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 30
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 31
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 32
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 33
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 34
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 35
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 36
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 37
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 38
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 39
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 40
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 41
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 42
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 43
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 44
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 45
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 46
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 47
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 48
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 49
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 50
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 51
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 52
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 53
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 54
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 55
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 56
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 57
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 58
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 59
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 60
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 61
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 62
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 63
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 64
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 65
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 66
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 67
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 68
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh

# spec_decoding_xlarge - 69
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=False SAMPLE=True LOG_SPEC_DECODE_METRICS=False ./run_sample.sh