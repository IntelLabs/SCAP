# Statistical Calibrated Activation Pruning (SCAP)

This repo contains the reference codes for _"Post-Training Statistical Calibration for Higher Activation Sparsity"_.

## Abstract

We present Statistical Calibrated Activation Pruning (SCAP), a post-training activation pruning framework that (1) generalizes sparsification by input activations of Fully-Connected layers for generic and flexible application across Transformers, and (2) features a simple Mode-Centering technique to pre-calibrate activation distributions for maximizing post-training sparsity. Our results demonstrate robust Pareto efficiency compared to prior methods, translating to a 1.5× additional LLM decoding speedup against CATS at iso model quality. SCAP effectiveness is empirically verified across a wide range of models, including recent Transformer Decoders, MoE, Mamba2, Encoding Transformer, and pre-quantized models, highlighting its practicality and scalability.

## Setup

Please follow the steps below.

```bash
# recommended python version: 3.10.13
python -m venv ./scap_env
source ./scap_env/bin/activate

# install torch
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# install dependencies
pip install transformers==4.44.0 datasets==2.21.0 accelerate tqdm rich seaborn matplotlib wheel \
    git+https://github.com/EleutherAI/lm-evaluation-harness.git@906ef948dc8dbb4c84e1bb0f2861b1aba30ab533

# install gemv kernel
pip install triton "git+https://github.com/ScalingIntelligence/CATS.git@0bda7708b835f20c59f4dd59d3d32b0c5f2f6376#egg=flash_gemv&subdirectory=flash_gemv"
```

## Reproducing the results

### 1. Run calibration

Get the calibrated thresholds of SCAP for each model and sparsity config.

```bash
bash scripts/01.calibration.bash
```

_You can skip this calibration step, as we have already uploaded the following model configs in the repo._

| Model ID                  | Config in the bash                         | Up/gate sparsity           | Down sparsity               |
| ------------------------- | ------------------------------------------ | -------------------------- | --------------------------- |
| meta-llama/Llama-2-7b-hf  | up,zero,0.35,gate,zero,0.35,down,zero,0.55 | 35% without mode centering | 55% without mode centering  |
| mistralai/Mistral-7B-v0.1 | up,zero,0.3,gate,zero,0.3,down,zero,0.7    | 30% without mode centering | 70% without mode centering  |
| mosaicml/mpt-7b           | down,kde,0.5                               | /                          | 50% with _kde peak_ as mode |
| tiiuae/falcon-7b          | down,median,0.5                            | /                          | 50% with _median_ as mode   |

The resulting `calibrated_thresholds.json` file at `results/scap/` folder shows the mode and threshold for each FFN layer specified in the config.

### 2. Evaluation on zero-shot tasks

Evaluate the zero-shot tasks listed in the paper, i.e., _winogrande, piqa, sciq, hellaswag, boolq, arc_easy, arc_challenge_.
Results are at `results/scap/` folder.

```bash
bash scripts/02.evaluate_zero_shot_tasks.bash
```

The resulting `evaluation_results.json` file contains: (1) evaluation metrics for each task; (2) averaged actual input sparsity for each layer.

### 3. Inference with sparse kernel

We show the actual inference of SCAP optimized models with the sparse GEMV kernel.

```bash
bash scripts/03.inference_demo.bash
```

## Acknowledgement

This work is built atop [CATS](https://github.com/ScalingIntelligence/CATS), which we believe also extends from [DejaVu](https://github.com/FMInference/DejaVu). Credits go to the original authors of these projects.
