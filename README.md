# Speculative Decoding for ProGen2


## Install

```
git clone git@github.com:amyxlu/progen-speculative-decoding.git
cd progen-speculative-decoding
pip install -e .
pip install -r requirements.txt   # modified to be more compatible w/ more recent package versions
```

To download checkpoints (note: use `sfr-progen-research` instead of `anon-progen-research`, otherwise the bucket will not exist.)
```
model=progen2-small
wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
tar -xvf /data/fjiahai/checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/
```

Repeat for `model=progen2-xlarge`.

## Basic Generation

With vllm (skip sanity check):

```
python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=True --sanity=False
```

Without vllm:

```
python sample.py --model progen2-xlarge --num-samples 1 --max-length 512 --use_vllm=False
```

with ragged batches:
```
python sample.py --fp16 False --ragged-batches true --model progen2-xlarge
```

## Run modes

`sample.py` provides four main run modes:

- `--sanity`: sanity check that the model cross-entropy is correct on a test sequence. NOTE: this does not currently work with vllm.
- `--sample`: whether to sample from the model.
- `--benchmark`: whether to run the timing benchmark.
- `--log_spec_decode_metrics`: whether to log speculative decoding metrics. This is mutually exclusive with `--sample=True` and `--benchmark=True`, and requires `--use_vllm=True`.

## Sampling with Speculative Decoding
```
python run_speculative_sampling.py \
  --draft_model progen2-small \
  --target_model progen2-xlarge \
  --num-reruns 8 \
  --max-length 512
```

With ragged batches. This will also automatically run batched speculative decoding.
```
python run_speculative_sampling.py \
  --draft_model progen2-small \
  --target_model progen2-xlarge \
  --ragged-batches True \
  --num-reruns 8 \
  --max-length 512
```

# ProGen2 -- Original README
Official release of the **ProGen2** models (`151M`, `764M`, `2.7B`, `6.4B`) for **Protein Engineering**.

## Models

| Model | Size | Checkpoint |
| ------ | ------ | ---------- |
| progen2-small	   | `151M` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-small.tar.gz |
| progen2-medium   | `764M` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-medium.tar.gz |
| progen2-oas	     | `764M` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-oas.tar.gz |
| progen2-base     | `764M` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-base.tar.gz |
| progen2-large    | `2.7B` |  https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-large.tar.gz |
| progen2-BFD90    | `2.7B` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-BFD90.tar.gz |
| progen2-xlarge   | `6.4B` | https://storage.googleapis.com/anon-progen-research/checkpoints/progen2-xlarge.tar.gz |

## Setup
```sh
# code
git clone https://github.com/anon-progen-research/progen
cd progen2

# checkpoint
model=progen2-large
wget -P checkpoints/${model} https://storage.googleapis.com/anon-progen-research/checkpoints/${model}.tar.gz
tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/

# venv
python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# sample
python3 sample.py --model ${model} --t 0.8 --p 0.9 --max-length 1024 --num-samples 2 --context "1"

# log-likelihood (GenBank: TMF32756.1)
python3 likelihood.py --model ${model} --context "1MGHGVSRPPVVTLRPAVLDDCPVLWRWRNDPETRQASVDEREIPVDTHTRWFEETLKRFDRKLFIVSADGVDAGMVRLDIQDRDAAVSVNIAPEWRGRGVGPRALGCLSREAFGPLALLRMSAVVKRENAASRIAFERAGFTVVDTGGPLLHSSKARLHVVAAIQARMGSTRLPGKVLVSIAGRPTIQRIAERLAVCQELDAVAVSTSVENRDDAIADLAAHLGLVCVRGSETDLIERLGRTAARTGADALVRITADCPLVDPALVDRVVGVWRRSAGRLEYVSNVFPPTFPDGLDVEVLSRTVLERLDREVSDPFFRESLTAYVREHPAAFEIANVEHPEDLSRLRWTMDYPEDLAFVEAVYRRLGNQGEIFGMDDLLRLLEWSPELRDLNRCREDVTVERGIRGTGYHAALRARGQAP2"
```

## License
Our code and models are BSD-3 licensed. See LICENSE.txt for details.

## Ethics
Predicting the fitness of a protein sequence and capturing the distribution of natural proteins for generative purposes could be a powerful tool for protein design. If our technique or a future iteration thereof is adopted broadly, care should be taken in terms of the end use-cases of these designed samples and downstream effects to ensure safe, non-nefarious, and ethical applications. For projects in any domain, active oversight during project initiation, experimental optimization, and deployment phases should be put in place to ensure safe usage and limitation of unintended harmful effects.
