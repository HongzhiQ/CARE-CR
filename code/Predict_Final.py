import os
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from typing import  Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict
from layers.Bert_Predictor import PreferencePredictor, load_tokenizer, resolve_checkpoint


class PreferenceScorer:
    def __init__(self, local_model_path=None, model_name=None, save_dir=None, checkpoint_path=None, max_length=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.tokenizer = load_tokenizer(save_dir, local_model_path, model_name)
        self.model = PreferencePredictor(model_name=model_name, local_model_path=local_model_path, num_labels=6)
        self.model.to(self.device)
        ckpt = resolve_checkpoint(save_dir, checkpoint_path)
        if ckpt:
            state = torch.load(ckpt, map_location=self.device)
            model_state = state.get('model_state_dict', state)
            current_state = self.model.state_dict()
            matching_state = {
                k: v
                for k, v in model_state.items()
                if k in current_state and hasattr(v, "shape") and current_state[k].shape == v.shape
            }
            if matching_state:
                self.model.load_state_dict(matching_state, strict=False)
        self.model.eval()

    def predict(self, text: str) -> List[float]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        with torch.no_grad():
            probs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return probs.squeeze(0).detach().cpu().numpy().tolist()


def predict(text, local_model_path=None, model_name=None, save_dir=None, checkpoint_path=None, max_length=512):
    scorer = PreferenceScorer(
        local_model_path=local_model_path,
        model_name=model_name,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
        max_length=max_length,
    )
    return scorer.predict(text)

def load_generation_tokenizer(base_model_name):
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok

def load_base_causal_model(base_model_name):
    mdl = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", trust_remote_code=True, local_files_only=True)
    return mdl

def _sanitize_path(p):
    from pathlib import Path
    try:
        return str(Path(p).resolve())
    except Exception:
        return p.replace("\\", "/")

def _find_adapter_dir(start_dir):
    must_have = {"adapter_config.json"}
    weight_files = {"adapter_model.bin", "adapter_model.safetensors"}
    if os.path.isdir(start_dir):
        files = set(os.listdir(start_dir))
        if must_have.issubset(files) and files.intersection(weight_files):
            return start_dir
    for root, files in os.walk(start_dir):
        s = set(files)
        if must_have.issubset(s) and s.intersection(weight_files):
            return root
    return None


def _load_adapter_state_dict(adapter_dir: str) -> Dict[str, torch.Tensor]:
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(bin_path):
        state = torch.load(bin_path, map_location="cpu")
        return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path)
        return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    raise FileNotFoundError(f"LoRA weight file could not be found: {adapter_dir}")


def _normalize_lora_key(key: str) -> str:
    k = str(key)
    k = k.replace(".default.", ".")
    k = k.replace("..", ".")
    return k


def _map_adapter_state_to_model_params(
    adapter_state: Dict[str, torch.Tensor],
    lora_param_name_by_norm: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    mapped: Dict[str, torch.Tensor] = {}
    norm_to_tensor: Dict[str, torch.Tensor] = {}
    for k, v in adapter_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        norm_to_tensor[_normalize_lora_key(k)] = v

    for norm_k, v in norm_to_tensor.items():
        dst = lora_param_name_by_norm.get(norm_k)
        if dst is not None:
            mapped[dst] = v

    if mapped:
        return mapped

    model_norm_keys = list(lora_param_name_by_norm.keys())
    for norm_k, v in norm_to_tensor.items():
        candidates = [mk for mk in model_norm_keys if mk.endswith(norm_k) or norm_k.endswith(mk)]
        if len(candidates) == 1:
            mapped[lora_param_name_by_norm[candidates[0]]] = v

    return mapped


class WeightedLoRAGenerator:
    def __init__(self, base_model_name: str, peft_names: List[str]):
        self.base_model_name = base_model_name
        self.peft_names = list(peft_names)
        self.base_model = load_base_causal_model(base_model_name)
        self.tokenizer = load_generation_tokenizer(base_model_name)

        resolved_dirs: Dict[str, str] = {}
        for name in self.peft_names:
            local_path = _sanitize_path(name)
            adapter_dir = _find_adapter_dir(local_path) or local_path
            resolved_dirs[name] = adapter_dir

        first_dir = resolved_dirs[self.peft_names[0]]
        self.model = PeftModel.from_pretrained(self.base_model, first_dir, local_files_only=True)
        self.model.eval()

        self.lora_params: Dict[str, torch.nn.Parameter] = {}
        self.lora_param_name_by_norm: Dict[str, str] = {}
        for n, p in self.model.named_parameters():
            if "lora_" in n:
                self.lora_params[n] = p
                self.lora_param_name_by_norm[_normalize_lora_key(n)] = n

        self.adapter_states: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self.peft_names:
            state = _load_adapter_state_dict(resolved_dirs[name])
            filtered = _map_adapter_state_to_model_params(state, self.lora_param_name_by_norm)
            self.adapter_states[name] = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in filtered.items()}

    def set_weights(self, peft_list: List[str], coefficients: List[float]) -> None:
        coeff_map = {n: float(c) for n, c in zip(peft_list, coefficients)}
        keys = set()
        for n in peft_list:
            keys.update(self.adapter_states[n].keys())

        with torch.no_grad():
            for k in keys:
                acc = None
                for n in peft_list:
                    c = coeff_map.get(n, 0.0)
                    if c == 0.0:
                        continue
                    v = self.adapter_states[n].get(k)
                    if v is None:
                        continue
                    if acc is None:
                        acc = v.mul(c)
                    else:
                        acc = acc.add(v, alpha=c)
                if acc is None:
                    continue
                p = self.lora_params.get(k)
                if p is None:
                    continue
                p.copy_(acc.to(device=p.device, dtype=p.dtype))

    def generate(self, text: str, max_new_tokens=256, temperature=0.8, top_p=0.9) -> str:
        return generate_response(
            self.model,
            self.tokenizer,
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

def average_lora_weights(base_model, peft_names, coefficients):
    weights_averaged = {}
    total = 0
    for peft_name, coeff in zip(peft_names, coefficients):
        if coeff == 0:
            continue
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("LOCAL_FILES_ONLY", "1")
        local_path = _sanitize_path(peft_name)
        adapter_dir = _find_adapter_dir(local_path)
        if adapter_dir is None:
            print(f"Trying to load the path: {local_path}")
            adapter_dir = local_path
        current_model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
        current_weights = get_peft_model_state_dict(current_model, state_dict=None)
        for k, v in list(current_weights.items()):
            if total == 0 and k not in weights_averaged:
                weights_averaged[k] = coeff * v
            else:
                weights_averaged[k] = weights_averaged.get(k, 0) + coeff * v
            del current_weights[k]
        del current_model
        torch.cuda.empty_cache()
        total += 1
    return weights_averaged

def build_weight_averaged_model(base_model_name, peft_names, coefficients):
    base_model = load_base_causal_model(base_model_name)
    wa_state = average_lora_weights(base_model, peft_names, coefficients)
    adapter_dir0 = _find_adapter_dir(_sanitize_path(peft_names[0])) or _sanitize_path(peft_names[0])
    wa_model = PeftModel.from_pretrained(base_model, adapter_dir0, local_files_only=True)
    wa_model.load_state_dict(wa_state, strict=False)
    wa_model.eval()
    tok = load_generation_tokenizer(base_model_name)
    return wa_model, tok

def generate_response(model, tokenizer, text, max_new_tokens=256, temperature=0.8, top_p=0.9):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys_prompt = (
        "<system prompt>"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    enc = tokenizer(prompt_text, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attn = enc.get('attention_mask', None)
    if attn is not None:
        attn = attn.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    gen_ids = out[0][input_ids.shape[1]:]
    text_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text_out.strip()


def _resolve_path(p: str) -> Path:
    path = Path(str(p).strip())
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def _default_output_csv(input_csv: Path) -> Path:
    if input_csv.suffix.lower() != ".csv":
        return input_csv.with_name(input_csv.name + "_pred.csv")
    return input_csv.with_name(input_csv.stem + "_pred.csv")


def _iter_first_column_texts(input_csv: Path) -> List[str]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"The header of the CSV file was not found: {input_csv}")
        first_col = fieldnames[0]
        texts: List[str] = []
        for row in reader:
            if not row:
                continue
            text = str(row.get(first_col, "")).strip()
            if not text:
                continue
            texts.append(text)
    return texts

def main():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--input_csv', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--local_model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--base_model_name', type=str, default=None)
    default_peft_names = [
        "The path of the Expert Strategy Model 1",
        "The path of the Expert Strategy Model 2",
        "The path of the Expert Strategy Model 3",
        "The path of the Expert Strategy Model 4",
        "The path of the Expert Strategy Model 5",
        "The path of the Expert Strategy Model 6",

    ]
    parser.add_argument('--peft_names', type=str, nargs='*', default=default_peft_names)
    parser.add_argument('--gen_max_new_tokens', type=int, default=None)
    parser.add_argument('--gen_temperature', type=float, default=None)
    parser.add_argument('--gen_top_p', type=float, default=None)
    args = parser.parse_args()

    dims = ['同理心', '积极性', '理性', '可执行性', '具体性', '可读性']

    text = str(args.text).strip()
    input_csv_arg = str(args.input_csv).strip()
    peft_names = [p for p in (args.peft_names or []) if str(p).strip()]


    def pick_peft_list(coeffs_vec: np.ndarray) -> Tuple[List[str], np.ndarray]:
        if len(peft_names) == len(coeffs_vec):
            return peft_names, coeffs_vec
        if len(peft_names) > len(coeffs_vec):
            if len(peft_names) == len(coeffs_vec) + 1:
                mapped = peft_names[: len(coeffs_vec) - 1] + [peft_names[-1]]
                return mapped, coeffs_vec
            return peft_names[: len(coeffs_vec)], coeffs_vec
        mapped = peft_names
        trimmed = coeffs_vec[: len(mapped)]
        trimmed = trimmed / (trimmed.sum() + 1e-12)
        return mapped, trimmed

    scorer = PreferenceScorer(
        local_model_path=args.local_model_path,
        model_name=args.model_name,
        save_dir=args.save_dir,
        checkpoint_path=args.checkpoint_path,
        max_length=args.max_length,
    )
    generator = WeightedLoRAGenerator(args.base_model_name, peft_names)

    if not text and input_csv_arg:
        input_csv = _resolve_path(input_csv_arg)
        output_csv = _resolve_path(args.output_csv) if str(args.output_csv).strip() else _default_output_csv(input_csv)
        texts = _iter_first_column_texts(input_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", *dims])
            for x in texts:
                scores = scorer.predict(x)
                coeffs = np.asarray(scores, dtype=np.float32)
                coeffs = coeffs / (coeffs.sum() + 1e-12)
                peft_list, coeffs = pick_peft_list(coeffs)
                generator.set_weights(peft_list, coeffs.tolist())
                y = generator.generate(
                    x,
                    max_new_tokens=args.gen_max_new_tokens,
                    temperature=args.gen_temperature,
                    top_p=args.gen_top_p,
                )
                writer.writerow([x, y, *[f"{float(v):.6f}" for v in scores]])
        print(str(output_csv))
        return

    scores = scorer.predict(text)
    for dim, s in zip(dims, scores):
        print(f'{dim}: {s:.6f}')

    coeffs = np.asarray(scores, dtype=np.float32)
    coeffs = coeffs / (coeffs.sum() + 1e-12)
    peft_list, coeffs = pick_peft_list(coeffs)
    generator.set_weights(peft_list, coeffs.tolist())
    y = generator.generate(
        text,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
    )
    print(y)

if __name__ == '__main__':
    main()
