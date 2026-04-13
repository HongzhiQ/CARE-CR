import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


@dataclass(frozen=True)
class ScriptArgs:
    adapter: str
    output_dir: str
    base_model: Optional[str]
    torch_dtype: str
    device_map: Optional[str]
    safe_serialization: bool
    max_shard_size: str
    trust_remote_code: bool
    seed: int


def _resolve_dtype(name: str):
    if name == 'auto':
        return None
    return {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[name]


def _parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, default=None)

    parser.add_argument('--torch_dtype', type=str, default='auto', choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--device_map', type=str, default=None)
    parser.add_argument('--safe_serialization', type=str, default='true')
    parser.add_argument('--max_shard_size', type=str, default='5GB')
    parser.add_argument('--trust_remote_code', type=str, default='true')
    parser.add_argument('--seed', type=int, default=42)

    ns = parser.parse_args()
    safe_serialization = str(ns.safe_serialization).lower() in {'1', 'true', 'yes', 'y', 't'}
    trust_remote_code = str(ns.trust_remote_code).lower() in {'1', 'true', 'yes', 'y', 't'}
    device_map = None if ns.device_map in {None, '', 'none', 'null'} else ns.device_map
    base_model = None if ns.base_model in {None, '', 'none', 'null'} else ns.base_model

    return ScriptArgs(
        adapter=ns.adapter,
        output_dir=ns.output_dir,
        base_model=base_model,
        torch_dtype=ns.torch_dtype,
        device_map=device_map,
        safe_serialization=safe_serialization,
        max_shard_size=ns.max_shard_size,
        trust_remote_code=trust_remote_code,
        seed=ns.seed,
    )


def _check_tie_word_embeddings(model) -> None:
    try:
        from peft.utils import ModulesToSaveWrapper
    except Exception:
        return
    config = getattr(model, 'config', None)
    if config is None:
        return
    if not bool(getattr(config, 'tie_word_embeddings', False)):
        return
    input_emb = getattr(model, 'get_input_embeddings', None)
    output_emb = getattr(model, 'get_output_embeddings', None)
    if not callable(input_emb) or not callable(output_emb):
        return
    if not isinstance(input_emb(), ModulesToSaveWrapper):
        return
    if not isinstance(output_emb(), ModulesToSaveWrapper):
        return
    config.tie_word_embeddings = False


def _infer_base_model_from_adapter(adapter_path: str, *, trust_remote_code: bool) -> str:
    try:
        from peft import PeftConfig
    except Exception as e:
        raise RuntimeError('peft is required for LoRA merge, please install peft') from e
    cfg = PeftConfig.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)
    base = getattr(cfg, 'base_model_name_or_path', None)
    if not base:
        raise ValueError('Cannot infer base_model_name_or_path from adapter config; please pass --base_model')
    return str(base)


def _maybe_resize_token_embeddings(model, tokenizer) -> None:
    vocab_size = getattr(getattr(model, 'config', None), 'vocab_size', None)
    if vocab_size is None:
        return
    if len(tokenizer) <= int(vocab_size):
        return
    if hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(len(tokenizer))


def merge_lora(adapter_path: str,
               output_dir: str,
               *,
               base_model: Optional[str],
               torch_dtype: str,
               device_map: Optional[str],
               safe_serialization: bool,
               max_shard_size: str,
               trust_remote_code: bool) -> None:
    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError('peft is required for LoRA merge, please install peft') from e

    resolved_base_model = base_model or _infer_base_model_from_adapter(adapter_path, trust_remote_code=trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        torch_dtype=_resolve_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    _maybe_resize_token_embeddings(model, tokenizer)

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    _check_tie_word_embeddings(model)
    merged = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(output_dir)

    generation_config = getattr(merged, 'generation_config', None)
    if generation_config is not None and hasattr(generation_config, 'save_pretrained'):
        generation_config.save_pretrained(output_dir)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    merge_lora(
        args.adapter,
        args.output_dir,
        base_model=args.base_model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == '__main__':
    main()

