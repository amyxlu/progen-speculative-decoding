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

### Baseline: progen2-small, medium, and xlarge for different NUM_SAMPLES and MAX_LENGTH.

# baseline - 0
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 1
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 2
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 3
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 4
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 5
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 6
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 7
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 8
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 9
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 10
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 11
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 12
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 13
MODEL=progen2-small NUM_SAMPLES=4 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 14
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 15
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 16
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 17
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 18
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 19
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 20
MODEL=progen2-small NUM_SAMPLES=16 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 21
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 22
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 23
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 24
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 25
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 26
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 27
MODEL=progen2-small NUM_SAMPLES=64 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 28
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 29
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 30
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 31
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 32
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 33
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 34
MODEL=progen2-small NUM_SAMPLES=128 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 35
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 36
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 37
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 38
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 39
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 40
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 41
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 42
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 43
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 44
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 45
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 46
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 47
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 48
MODEL=progen2-medium NUM_SAMPLES=4 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 49
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 50
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 51
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 52
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 53
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 54
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 55
MODEL=progen2-medium NUM_SAMPLES=16 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 56
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 57
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 58
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 59
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 60
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 61
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 62
MODEL=progen2-medium NUM_SAMPLES=64 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 63
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 64
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 65
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 66
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 67
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 68
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 69
MODEL=progen2-medium NUM_SAMPLES=128 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 70
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 71
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 72
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 73
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 74
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 75
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 76
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 77
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 78
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 79
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 80
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 81
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 82
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 83
MODEL=progen2-xlarge NUM_SAMPLES=4 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 84
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 85
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 86
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 87
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 88
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 89
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 90
MODEL=progen2-xlarge NUM_SAMPLES=16 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 91
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 92
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 93
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 94
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 95
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 96
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 97
MODEL=progen2-xlarge NUM_SAMPLES=64 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 98
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 99
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 100
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 101
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 102
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 103
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline - 104
MODEL=progen2-xlarge NUM_SAMPLES=128 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### baseline_longer_max_len: same as baseline, but for longer MAX_LENGTH.
# baseline_longer_max_len - 0
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 1
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 2
MODEL=progen2-small NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 3
MODEL=progen2-small NUM_SAMPLES=8 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 4
MODEL=progen2-small NUM_SAMPLES=8 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 5
MODEL=progen2-small NUM_SAMPLES=8 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 6
MODEL=progen2-small NUM_SAMPLES=32 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 7
MODEL=progen2-small NUM_SAMPLES=32 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 8
MODEL=progen2-small NUM_SAMPLES=32 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 9
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 10
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 11
MODEL=progen2-medium NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 12
MODEL=progen2-medium NUM_SAMPLES=8 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 13
MODEL=progen2-medium NUM_SAMPLES=8 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 14
MODEL=progen2-medium NUM_SAMPLES=8 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 15
MODEL=progen2-medium NUM_SAMPLES=32 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 16
MODEL=progen2-medium NUM_SAMPLES=32 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 17
MODEL=progen2-medium NUM_SAMPLES=32 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 18
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 19
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 20
MODEL=progen2-xlarge NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 21
MODEL=progen2-xlarge NUM_SAMPLES=8 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 22
MODEL=progen2-xlarge NUM_SAMPLES=8 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 23
MODEL=progen2-xlarge NUM_SAMPLES=8 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 24
MODEL=progen2-xlarge NUM_SAMPLES=32 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 25
MODEL=progen2-xlarge NUM_SAMPLES=32 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# baseline_longer_max_len - 26
MODEL=progen2-xlarge NUM_SAMPLES=32 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with xlarge target model for different NUM_SPEC_TOKENS and MAX_LENGTH. NUM_SAMPLES=1.

# spec_decoding_xlarge - 0
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 1
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 2
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 3
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 4
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 5
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 6
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 7
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 8
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 9
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 10
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 11
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 12
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 13
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 14
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 15
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 16
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 17
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 18
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 19
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 20
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 21
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 22
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 23
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 24
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 25
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 26
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 27
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 28
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 29
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 30
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 31
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 32
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 33
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 34
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 35
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 36
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 37
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 38
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 39
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 40
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 41
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 42
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 43
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 44
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 45
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 46
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 47
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 48
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 49
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 50
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 51
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 52
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 53
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 54
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 55
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 56
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 57
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 58
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 59
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 60
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 61
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 62
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 63
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 64
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 65
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 66
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 67
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 68
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 69
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 70
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 71
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 72
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 73
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 74
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 75
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 76
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 77
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 78
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 79
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 80
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 81
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 82
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 83
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 84
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 85
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 86
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 87
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 88
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 89
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 90
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 91
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 92
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 93
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 94
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 95
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 96
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 97
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 98
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 99
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 100
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 101
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 102
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 103
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 104
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 105
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 106
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 107
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 108
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 109
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 110
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 111
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 112
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 113
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 114
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 115
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 116
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 117
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 118
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 119
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 120
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 121
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 122
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 123
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 124
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 125
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 126
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 127
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 128
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 129
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 130
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 131
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 132
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 133
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 134
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 135
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 136
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 137
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 138
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge - 139
MODEL=progen2-xlarge SPEC_MODEL=progen2-medium NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with medium target model for different NUM_SPEC_TOKENS and MAX_LENGTH. NUM_SAMPLES=1.

# spec_decoding_medium - 0
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 1
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 2
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 3
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 4
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 5
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 6
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 7
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 8
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 9
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 10
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 11
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 12
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 13
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 14
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 15
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 16
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 17
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 18
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 19
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 20
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 21
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 22
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_medium - 23
MODEL=progen2-medium SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with n-gram target "model" (not an ML model) for different NUM_SPEC_TOKENS and MAX_LENGTH. NUM_SAMPLES=1.

# spec_decoding_ngram - 0
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 1
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 2
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 3
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 4
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 5
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 6
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 7
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 8
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 9
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 10
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 11
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 12
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 13
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 14
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 15
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 16
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 17
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 18
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 19
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 20
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 21
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 22
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 23
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 24
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 25
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 26
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 27
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 28
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 29
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 30
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 31
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 32
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 33
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 34
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 35
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 36
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 37
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 38
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 39
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 40
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 41
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 42
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 43
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 44
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 45
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 46
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 47
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 48
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 49
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 50
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 51
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 52
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 53
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 54
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 55
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 56
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 57
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 58
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 59
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 60
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 61
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 62
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 63
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 64
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 65
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 66
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 67
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 68
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 69
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 70
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 71
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 72
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 73
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 74
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 75
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 76
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 77
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 78
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 79
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 80
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 81
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 82
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 83
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 84
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 85
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 86
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 87
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 88
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 89
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 90
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 91
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 92
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 93
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 94
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 95
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 96
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 97
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 98
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 99
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 100
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 101
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 102
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 103
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 104
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 105
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 106
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 107
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 108
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 109
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 110
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 111
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 112
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 113
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 114
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 115
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 116
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 117
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 118
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 119
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 120
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 121
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 122
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 123
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 124
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 125
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 126
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 127
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 128
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 129
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 130
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 131
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 132
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 133
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 134
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 135
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 136
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 137
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 138
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 139
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 140
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 141
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 142
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 143
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 144
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 145
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 146
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 147
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 148
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 149
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 150
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 151
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 152
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 153
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 154
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 155
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 156
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 157
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 158
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 159
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 160
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 161
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 162
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 163
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 164
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 165
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 166
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 167
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 168
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 169
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 170
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 171
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 172
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 173
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 174
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 175
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 176
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 177
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 178
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 179
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 180
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 181
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 182
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 183
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 184
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 185
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 186
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 187
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 188
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 189
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 190
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 191
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 192
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 193
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 194
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 195
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 196
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 197
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 198
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 199
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 200
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 201
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 202
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 203
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=64 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 204
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=128 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 205
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 206
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=512 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 207
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=1024 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 208
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=2048 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram - 209
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=4096 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with n-gram target "model" for longer MAX_LENGTH. NUM_SAMPLES=1, MAX_LENGTH=256.

# spec_decoding_ngram_longer_draft_seq_len - 0
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=16 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 1
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=32 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 2
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=64 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 3
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=96 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 4
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=128 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 5
MODEL=progen2-small SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=192 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 6
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=16 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 7
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=32 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 8
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=64 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 9
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=96 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 10
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=128 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 11
MODEL=progen2-medium SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=192 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 12
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=16 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 13
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=32 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 14
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=64 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 15
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=96 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 16
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=128 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_ngram_longer_draft_seq_len - 17
MODEL=progen2-xlarge SPEC_MODEL='[ngram]' NUM_SPEC_TOKENS=192 NUM_SAMPLES=1 MAX_LENGTH=256 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with small target model for different TEMP and NUM_SPEC_TOKENS values. NUM_SAMPLES=1, MAX_LENGTH=256.

# spec_decoding_xlarge_temp - 0
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 1
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 2
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 3
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 4
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 5
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 6
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 7
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 8
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 9
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 10
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 11
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 12
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 13
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 14
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 15
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 16
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 17
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 18
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 19
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 20
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 21
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 22
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 23
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 24
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 25
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 26
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 27
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 28
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 29
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 30
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 31
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 32
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 33
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 34
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 35
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 36
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 37
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 38
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 39
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 40
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 41
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 42
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 43
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 44
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 45
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 46
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 47
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 48
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 49
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 50
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 51
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 52
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 53
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 54
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 55
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.01 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 56
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 57
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 58
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_temp - 59
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TEMP=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with small target model for different TOP_P values and NUM_SPEC_TOKENS. NUM_SAMPLES=1, MAX_LENGTH=256.

# spec_decoding_xlarge_top_p - 0
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 1
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 2
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 3
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 4
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 5
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 6
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 7
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 8
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 9
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 10
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 11
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 12
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 13
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 14
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 15
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 16
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 17
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 18
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 19
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 20
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 21
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 22
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 23
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 24
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 25
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 26
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 27
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.8 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 28
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=0.9 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_top_p - 29
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 TOP_P=1.0 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

### Speculative decoding with small target model for different FREQ_PENALTY values and NUM_SPEC_TOKENS. NUM_SAMPLES=1, MAX_LENGTH=256.

# spec_decoding_xlarge_freq_penalty - 0
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 1
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 2
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=1 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 3
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 4
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 5
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=2 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 6
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 7
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 8
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=3 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 9
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 10
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 11
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=4 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 12
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 13
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 14
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=5 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 15
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 16
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 17
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=6 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 18
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 19
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 20
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=7 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 21
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 22
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 23
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=8 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 24
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 25
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 26
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=9 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 27
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.1 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 28
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.2 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh

# spec_decoding_xlarge_freq_penalty - 29
MODEL=progen2-xlarge SPEC_MODEL=progen2-small NUM_SPEC_TOKENS=10 NUM_SAMPLES=1 MAX_LENGTH=256 FREQ_PENALTY=0.5 BENCHMARK=True LOG_SPEC_DECODE_METRICS=False ./slurm_sample.sh
