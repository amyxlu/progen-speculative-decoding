# Speculative Decoding for ProGen2


## Install

```
git clone git@github.com:amyxlu/progen-speculative-decoding.git
cd progen-speculative-decoding
pip install -e .
```

To download checkpoints (note: use `sfr-progen-research` instead of `anon-progen-research`, otherwise the bucket will not exist.)
```
model=progen2-small
wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/
```

Repeat for `model=progen2-xlarge`.

## Basic Generation
```
cd progen-speculative-decoding
python sample.py --model progen2-xlarge --num-samples 8 --max-length 512
```

## Sampling with Speculative Decoding

TODO


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
