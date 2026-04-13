# CARE-CR
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow.svg)](https://huggingface.co/docs/transformers)

The code for the paper: **“CARE-CR: Context-Aware Routing and Expert Fusion for Multi-Preference Cognitive Restructuring”**. 

CARE-CR studies clinician-aligned cognitive restructuring under **context-dependent multi-preference control** by learning 
- **dimension-wise reward models** from pairwise preferences, 
- a **contextual preference predictor** that outputs simplex weights over clinical dimensions, 
- **Dimension-Guided Hierarchical MCTS (DG-HMCTS)** for structured search and synthetic preference augmentation, and 
- **expert fusion / dimension-specific policies** via preference optimization (e.g., DPO).

## Highlights
- Contextual preference prediction on a simplex (`λ_ctx(x)`) to represent clinician priorities.
- DG-HMCTS over a 3-level cognitive-reframing strategy tree (E→C→S) with dimension-guided priors.
- Dimension-wise reward models (`r_k`) and dimension-specific policy experts (`π_{θ_k}`).
- Script-first training workflow with minimal repository coupling.

## Table of Contents
- [Dataset](#dataset)
- [Data Formats](#data-formats)
- [Method](#method)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Reproducibility](#reproducibility)
- [Ethical Considerations](#ethical-considerations)

## Dataset

### Cognitive restructuring dataset
This repository also includes a cognitive restructuring dataset, denoted as **$D_{cog}$**, for supervised fine-tuning (SFT) of the base policy model.

- Location: `data/Cognitive Reconfiguration Dataset for SFT/` (`train.csv`, `test.csv`)
- Usage: `train.csv` for SFT; `test.csv` for held-out evaluation across models

Columns:

- `text_content`: the original user input (raw context)
- `comment`: the expert-annotated cognitive restructuring response
- `A1–A5`, `B1–B10`, `C1–C2`, `D1–D2`: cognitive-path labels
- `A`: triggering event
- `B`: cognitive distortion label(s)
- `C`: consequence(s) (e.g., emotional/behavioral effects)
- `D`: rebuttal cues expressed in the user utterance

### Expert-labeled preference table
 The expert-labeled preference table `data/Expert-labeled Dateset/CBT_2way_6dims.csv` supports two supervision signals used throughout the pipeline :

- **Dimension-wise preference data $D_{pref}^k$ (for reward models and preference optimization)**  
  Each record contains an input `x` and two candidate cognitive restructuring responses (`候选 A（重构）`, `候选 B（重构）`). Experts provide per-dimension scores for both candidates (columns `D2–D7`, each with `A/B` subcolumns) and an overall selection. From this table, one can construct `D_pref^(k) = {(x, y^+ ≻_k y^-)}` by comparing candidate A vs. B under dimension `k` (ties can be discarded or handled with margins).

- **Context–preference supervision $D_{\lambda}$ (for training the preference predictor $f_\phi$)**  
  The last columns provide an expert-assessed simplex weight vector `λ_true(x)` over the 6 dimensions (Empathy/Positivity/Rationality/Actionability/Specificity/Readability; corresponding to `D2–D7` in the CSV). These weights form `D_λ = {(x, λ_true(x))}`, used to train the contextual preference predictor that outputs `λ_ctx(x)`.

Note: The CSV includes an additional dimension `D1 思维陷阱`. The current codebase operationalizes **6 preference dimensions** and uses the weight labels aligned with `D2–D7`.

### DG-MCTS–augmented preference pairs

Our DG-MCTS–augmented preference pairs are generated based on the **C2D2** dataset
(Findings of EMNLP 2023):
https://github.com/bcwangavailable/C2D2-Cognitive-Distortion

Due to the access restrictions of C2D2, we do not redistribute the underlying data
or any derived text.  
Instead, we release the DG-MCTS generation code and configuration**, which allows
researchers with authorized access to C2D2 to fully reproduce** the preference
pairs used in our experiments.


## Data Formats

### (A) Reward model training data (pairwise preferences)

`code/Train_RewardModel.py` supports JSON/JSONL datasets with either schema:

- Chat-style:

```json
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "rejected_response": "...",
  "margin": 0.0
}
```

### (B) Preference predictor training data (simplex supervision)

`code/Train_PreferencePredictor.py` expects a CSV with:
- `x`: input context text
- the 6 dimension columns listed in the table above (weights do not need to be perfectly normalized; the script normalizes if needed)

### (C) DG-HMCTS sampling input

`code/Train_MCTS.py` reads a `.csv` or `.tsv` table with a text field named `x` or `text` . 

### (D) Base policy SFT training data

`code/Train_SFT.py` expects a JSON file (loaded via `datasets.load_dataset('json', ...)`). Each record can use one of the supported schemas:
- `{"messages": [{"role": "...", "content": "..."}, ...]}`
- `{"prompt": "...", "response": "..."}`
- `{"instruction": "...", "input": "...", "output": "..."}`
- `{"text": "..."}`

## Method

### Notation
- Base policy: `π_{θ0}`
- Dimensions: `K = {1, …, |K|}` (this repo uses `|K| = 6`)
- Dimension-wise preference data: `D_pref^(k)` (pairwise preferences per dimension)
- Context–preference supervision: `D_λ = {(x, λ_true(x))}`, `λ(x)` lies on a simplex
- Unlabeled contexts: `D_x` for DG-HMCTS sampling

### Dimensions (implementation)

The code uses Chinese column names for historical compatibility, while the README uses their English descriptors:

| Index | English | CSV Column |
|---:|---|---|
| 1 | Empathy | 同理心 |
| 2 | Positivity | 积极性 |
| 3 | Rationality | 理性 |
| 4 | Actionability | 可执行性 |
| 5 | Specificity | 具体性 |
| 6 | Readability | 可读性 |

### Training Pipeline

1. **Train dimension reward models (per dimension)**  
   For each `k ∈ K`, train a reward model `r_k(x, y)` using a pairwise ranking loss on `D_pref^(k)`. Freeze reward parameters after training.

2. **Train the contextual preference predictor**  
   Train `f_ϕ(x)` on `D_λ` to predict simplex weights `λ_ctx(x)` (softmax head on an encoder).

3. **DG-HMCTS-based data augmentation (per context)**  
   For each `x ∈ D_x`, predict `λ_ctx(x)`, compute action priors by aligning `λ_ctx(x)` with each action profile `g(a)`, and run hierarchical MCTS over the 3-level strategy tree (Emotional opening E → Cognitive challenge C → Summarization/closure S). During simulation, generate `y` from `π_{θ0}`, evaluate rewards with `r_k`, and backpropagate along the visited path to collect a candidate set `Y(x)`.

4. **Construct synthetic preference pairs per dimension**  
   For each `k ∈ K`, score candidates `y ∈ Y(x)` using `r_k(x, y)`, then select `(y^+, y^-)` to form `D_aug^(k)`.

5. **Train dimension-specific expert policies via DPO**  
   For each `k ∈ K`, initialize `π_{θ_k} ← π_{θ0}`, build `D_pref,all^(k) = D_pref^(k) ∪ D_aug^(k)`, and fine-tune using DPO.

**Outputs**: preference predictor `f_ϕ` and a set of expert policies `{π_{θ_k}}_{k=1}^{|K|}`.

## Repository Layout

```
CARE-CR/
  code/
    Train_SFT.py
    Train_RewardModel.py
    Train_PreferencePredictor.py
    Train_MCTS.py
    Train_DPO.py
    Predict_PreferencePredictor.py
    Predict_Final.py
    Merge_Lora_And_BaseModel.py
  data/
    Cognitive Restructuring Dataset/
    Expert-labeled Dateset/
  layers/
    DG_HMCTS.py
    Bert_Predictor.py
  README.md
```

## Installation

Create a Python 3.10+ environment and install common research dependencies:

```bash
pip install torch transformers datasets peft numpy pandas scikit-learn matplotlib tqdm
```

This repository frequently loads models with `local_files_only=True`. If you run offline, ensure all required checkpoints/tokenizers exist locally.

## Running the Pipeline

### 1) (Optional) Supervised fine-tune the base policy (SFT)

Train `π_{θ0}` on `D_cog`-derived SFT data (JSON) before preference optimization:

```bash
python code/Train_SFT.py ^
  --model <base_causal_lm_dir> ^
  --dataset <sft_dataset.json> ^
  --output_dir <sft_output_dir>
```

### 2) Train a reward model (repeat per dimension)

```bash
python code/Train_RewardModel.py ^
  --model <base_or_init_reward_model> ^
  --dataset <pairwise_dataset.jsonl_or_json> ^
  --output_dir <output_dir>
```

### 3) Train the contextual preference predictor

```bash
python code/Train_PreferencePredictor.py ^
  --csv_path <preference_supervision.csv> ^
  --save_dir <output_dir>
```

The predictor checkpoint resolver looks for `best_model.pt` first, then `final_model.pt` under `--save_dir` (or you can provide an explicit checkpoint path).

### 4) Run DG-HMCTS sampling to produce candidates

`Train_MCTS.py` expects exactly `|K|` reward models (this repo uses `|K|=6`) and writes one JSON record per simulation result.

```bash
python code/Train_MCTS.py ^
  --input_csv <unlabeled_contexts.csv_or.tsv> ^
  --output_path <dg_hmcts_outputs.jsonl> ^
  --gen_model_path <generation_model_path> ^
  --reward_model_paths <rm1_path> <rm2_path> <rm3_path> <rm4_path> <rm5_path> <rm6_path> ^
  --pref_local_model_path <bert_encoder_dir> ^
  --pref_save_dir <preference_predictor_save_dir> ^
  --pref_checkpoint_path <preference_predictor_checkpoint> ^
  --pref_max_length 256 ^
  --num_simulations 32
```

Each JSON line includes:
- `input_text`: the context `x`
- `lambda_clin`: predicted preference weights `λ_ctx(x)`
- `candidate_text`: generated reframing candidate
- `path_codes`: DG-HMCTS action path (e.g., `["E2","C3","S1"]`)
- `dim_rewards`: per-dimension rewards `r_k(x, y)`

### 5) Build synthetic preference pairs per dimension

Convert DG-HMCTS records into DPO-style pairs by ranking candidates with `r_k` and selecting `(y^+, y^-)`. The exact filtering strategy is experiment-dependent.

### 6) Train a dimension-specific expert policy with DPO (repeat per dimension)

```bash
python code/Train_DPO.py ^
  --model <policy_init_model> ^
  --ref_model <reference_model_or_empty> ^
  --dataset <dpo_dataset.jsonl_or_json> ^
  --output_dir <output_dir>
```

### 7) Inference

End-to-end generation with preference-conditioned LoRA mixing:

- Single input:

```bash
python code/Predict_Final.py ^
  --text "<your input>" ^
  --save_dir <preference_predictor_save_dir> ^
  --checkpoint_path <preference_predictor_checkpoint> ^
  --local_model_path <bert_encoder_dir> ^
  --base_model_name <base_causal_lm_dir> ^
  --peft_names <expert1_lora_dir> <expert2_lora_dir> <expert3_lora_dir> <expert4_lora_dir> <expert5_lora_dir> <expert6_lora_dir> ^
  --gen_max_new_tokens 256 ^
  --gen_temperature 0.8 ^
  --gen_top_p 0.9
```

- Batch mode (CSV in → CSV out): the script reads the first column as input text and writes `x,y` plus 6 predicted weights.

```bash
python code/Predict_Final.py ^
  --input_csv <inputs.csv> ^
  --output_csv <outputs_pred.csv> ^
  --save_dir <preference_predictor_save_dir> ^
  --checkpoint_path <preference_predictor_checkpoint> ^
  --local_model_path <bert_encoder_dir> ^
  --base_model_name <base_causal_lm_dir> ^
  --peft_names <expert1_lora_dir> <expert2_lora_dir> <expert3_lora_dir> <expert4_lora_dir> <expert5_lora_dir> <expert6_lora_dir>
```

`code/Predict_Final.py` is project-specific wiring; integrating multiple expert policies depends on the modeling choice used in your experiments.

## Reproducibility
- Training scripts typically accept `--seed` (or use a default).
- Checkpoints are saved under the specified `output_dir`/`save_dir`.
- DG-HMCTS behavior depends on sampling hyperparameters (e.g., `num_simulations`, `c_puct`, `tau`, and layer-specific `beta_*`).

## Ethical Considerations

This repository is intended for research on clinically aligned generation and does **not** replace professional mental health services. If you handle sensitive user data, ensure compliance with privacy regulations and obtain appropriate ethics approvals.
