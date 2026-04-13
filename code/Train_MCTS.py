import csv
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from layers.DG_HMCTS import ACTION_DEFS, DG_HMCTS, DIM_NAMES, MCTSConfig, NUM_DIMS
from layers.Bert_Predictor import PreferencePredictor, load_tokenizer, resolve_checkpoint


def load_x_from_csv(csv_path: str) -> List[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            text = raw.get("x")
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            texts.append(text)
    return texts


def _onehot_is_positive(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    try:
        return float(s) > 0.5
    except Exception:
        return s.lower() in {"true", "yes", "y"}


def load_samples_from_table(input_path: str) -> List[Dict]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        text_key = None
        for k in ("x", "text"):
            if k in fieldnames:
                text_key = k
                break
        if text_key is None and fieldnames:
            text_key = fieldnames[0]

        label_cols = [
            c
            for c in fieldnames
            if c != text_key and str(c).strip()
        ]
        for raw in reader:
            if not raw:
                continue
            text = raw.get(text_key, "") if text_key else ""
            text = str(text).strip()
            if not text:
                continue
            labels_text = None
            if label_cols:
                active = [col for col in label_cols if _onehot_is_positive(raw.get(col))]
                labels_text = "、".join(active) if active else None
            samples.append({"text": text, "labels_text": labels_text})
    return samples


def build_lambda_predict_fn(
    local_model_path: str,
    model_name: str | None,
    save_dir: str | None,
    checkpoint_path: str | None,
    max_length: int = None,
) -> Callable[[str], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(save_dir, local_model_path, model_name)
    model = PreferencePredictor(model_name=model_name, local_model_path=local_model_path, num_labels=NUM_DIMS)
    model.to(device)
    ckpt = resolve_checkpoint(save_dir, checkpoint_path)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

    model.eval()

    def predict_lambda(text: str) -> List[float]:
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
        vec = probs.squeeze(0).cpu().numpy().astype(np.float32)
        s = float(vec.sum())
        if s > 0:
            vec = vec / s
        return vec.tolist()

    return predict_lambda



def run_stage2_for_dataset(
    csv_path: str,
    gen_fn: Callable[[str, List[str]], str],
    reward_fns: List[Callable[[str, str], float]],
    lambda_fn: Callable[[str], List[float]],
    num_simulations: int = 32,
    max_samples: int = None,
) -> List[Dict]:
    samples = load_samples_from_table(csv_path)
    total = len(samples)
    if max_samples is not None:
        total = min(total, max_samples)
    print(f"[Stage2] There are {total} samples")
    cfg = MCTSConfig(num_simulations=num_simulations)
    mcts = DG_HMCTS(cfg)
    results: List[Dict] = []
    for idx, sample in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break
        print(f"[Stage2] processing sample {idx + 1}/{total}")
        text = sample["text"]
        labels_text = sample.get("labels_text", None)

        def gen_fn_with_labels(t: str, path_codes: List[str]) -> str:
            try:
                return gen_fn(t, path_codes, labels_text=labels_text)
            except TypeError:
                return gen_fn(t, path_codes)

        lam_list = lambda_fn(text)
        lam = np.asarray(lam_list, dtype=np.float32)
        if lam.shape[0] != NUM_DIMS:
            continue
        s = float(lam.sum())
        if s <= 0:
            continue
        lam = lam / s
        sims = mcts.run(
            text=text,
            lambda_clin=lam.tolist(),
            gen_fn=gen_fn_with_labels,
            reward_fns=reward_fns,
            num_simulations=num_simulations,
        )
        for item in sims:
            record: Dict = {
                "input_text": text,
                "lambda_clin": lam.tolist(),
                "candidate_text": item["text"],
                "path_codes": item["path"],
                "dim_rewards": item["rewards"],
            }
            results.append(record)
        if (idx + 1) == total or ((idx + 1) % 10 == 0):
            print(f"[Stage2] Completed samples {idx + 1}/{total}")
    return results


def save_results_jsonl(records: List[Dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_gen_fn(gen_model_path: str | None = None) -> Callable[[str, List[str]], str]:
    if not gen_model_path:
        raise ValueError("gen_model_path cannot be null")
    tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        gen_model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model.eval()

    def gen_fn(text: str, path_codes: List[str], labels_text: str | None = None) -> str:
        device = model.device
        dim_text = "、".join(DIM_NAMES)
        step_lines: List[str] = []
        for i, code in enumerate(path_codes, start=1):
            info = ACTION_DEFS.get(code)
            desc = ""
            if info is not None and "description" in info:
                desc = str(info["description"])
            if code.startswith("E"):
                stage = "<Step 1>"
            elif code.startswith("C"):
                stage = "<Step 2>"
            elif code.startswith("S"):
                stage = "<Step 3>"

            if desc:
                step_lines.append(f"{stage}({code})Goals:{desc}")
            else:
                step_lines.append(f"{stage}({code})")
        step_plan = "\n".join(step_lines)
        sys_prompt = (
            """
            system: You are a helpful assistant.
            """
        )
        label_block = ""
        if labels_text is not None:
            label_block = f"The type of cognitive distortion of this user is:{labels_text}\n"  
        user_content = (
            "User input:\n"
            f"{text}\n"
            f"{label_block}"
            "Intervention strategy path:\n"
            f"{step_plan}\n"
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                do_sample=True,
                temperature=None,
                top_p=None,
                max_new_tokens=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][input_ids.shape[1]:]
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text_out.strip()

    return gen_fn


def build_reward_fns(
    reward_model_paths: List[str] | None = None,
    max_length: int = 2048,
) -> List[Callable[[str, str], float]]:
    if not reward_model_paths:
        raise ValueError("reward_model_paths cannot be null")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizers: List[AutoTokenizer] = []
    models: List[AutoModelForSequenceClassification] = []
    for p in reward_model_paths:
        tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            p,
            trust_remote_code=True,
            local_files_only=True,
        )
        mdl.to(device)
        mdl.eval()
        tokenizers.append(tok)
        models.append(mdl)
    fns: List[Callable[[str, str], float]] = []

    rm_system_prompt = (
        "Please output the reward score"
    )

    def make_fn(t, m):
        def fn(x: str, y: str) -> float:
            prompt_text = None
            try:
                messages = [
                    {"role": "system", "content": rm_system_prompt},
                    {"role": "user", "content": + x},
                    {"role": "assistant", "content": y},
                ]
                prompt_text = t.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                prompt_text = "user：" + x + "\n assistant：" + y
            enc = t(
                prompt_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
            with torch.no_grad():
                outputs = m(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                score_tensor = logits.squeeze(-1).mean()
            else:
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[-1] > 1:
                    score_tensor = probs[..., 1].mean()
                else:
                    score_tensor = probs.mean()
            score = float(score_tensor.detach().cpu().item())
            return score

        return fn

    for tok, mdl in zip(tokenizers, models):
        fns.append(make_fn(tok, mdl))
    return fns







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--generative_model_path", type=str, default=None)
    default_reward_names = [
        "The path of Reward Model 1",
        "The path of Reward Model 2",
        "The path of Reward Model 3",
        "The path of Reward Model 4",
        "The path of Reward Model 5",
        "The path of Reward Model 6",
    ]
    parser.add_argument("--reward_model_paths", type=str, nargs="*", default=default_reward_names)
    parser.add_argument("--Preference predictor_local_model_path", type=str, default=None)
    parser.add_argument("--Preference predictor_model_name", type=str, default=None)
    parser.add_argument("--Preference predictor_save_dir", type=str, default=None)
    parser.add_argument("--Preference predictor_checkpoint_path (optional)", type=str, default=None)
    parser.add_argument("--Preference predictor_max_length", type=int, default=None)
    parser.add_argument("--num_simulations", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()


    gen = build_gen_fn(args.gen_model_path)
    rewards = build_reward_fns(args.reward_model_paths)
    lambda_fn = build_lambda_predict_fn(
        local_model_path=args.pref_local_model_path,
        model_name=args.pref_model_name,
        save_dir=args.pref_save_dir,
        checkpoint_path=args.pref_checkpoint_path,
        max_length=args.pref_max_length,
    )

    out_records = run_stage2_for_dataset(
        csv_path=args.input_csv,
        gen_fn=gen,
        reward_fns=rewards,
        lambda_fn=lambda_fn,
        num_simulations=args.num_simulations,
        max_samples=args.max_samples,
    )
    save_results_jsonl(out_records, args.output_path)
