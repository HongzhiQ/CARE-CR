import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)


Messages = List[Dict[str, Any]]


@dataclass(frozen=True)
class ScriptArgs:
    model: str
    dataset: str
    output_dir: str
    max_length: int
    truncation_side: str
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    seed: int
    gradient_checkpointing: bool
    torch_dtype: str
    bf16: bool
    fp16: bool
    trust_remote_code: bool
    split_dataset_ratio: float
    max_train_samples: int
    max_eval_samples: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str


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
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--truncation_side', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    bool_action = getattr(argparse, 'BooleanOptionalAction', None)
    if bool_action is None:
        parser.add_argument('--gradient_checkpointing', type=str, default='true')
        parser.add_argument('--bf16', type=str, default='true')
        parser.add_argument('--fp16', type=str, default='false')
        parser.add_argument('--trust_remote_code', type=str, default='true')
    else:
        parser.add_argument('--gradient_checkpointing', action=bool_action, default=True)
        parser.add_argument('--bf16', action=bool_action, default=True)
        parser.add_argument('--fp16', action=bool_action, default=False)
        parser.add_argument('--trust_remote_code', action=bool_action, default=True)

    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--split_dataset_ratio', type=float, default=0.1)
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)

    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

    ns = parser.parse_args()
    if bool_action is None:
        ns.gradient_checkpointing = str(ns.gradient_checkpointing).lower() in {'1', 'true', 'yes', 'y', 't'}
        ns.bf16 = str(ns.bf16).lower() in {'1', 'true', 'yes', 'y', 't'}
        ns.fp16 = str(ns.fp16).lower() in {'1', 'true', 'yes', 'y', 't'}
        ns.trust_remote_code = str(ns.trust_remote_code).lower() in {'1', 'true', 'yes', 'y', 't'}
    return ScriptArgs(**vars(ns))


def _load_and_split_dataset(dataset_path: str, *, seed: int, split_ratio: float) -> Tuple[Dataset, Optional[Dataset]]:
    ds = load_dataset('json', data_files={'data': dataset_path})['data']
    ratio = max(0.0, min(0.99, float(split_ratio)))
    if ratio <= 0:
        return ds, None
    split = ds.train_test_split(test_size=ratio, seed=seed, shuffle=True)
    return split['train'], split['test']


def _apply_chat_template(tokenizer, messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    parts = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        parts.append(f'[{role}]\n{content}')
    if add_generation_prompt:
        parts.append('[assistant]\n')
    return '\n'.join(parts)


def _encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _truncate(input_ids: List[int], labels: List[int], max_length: int, truncation_side: str) -> Tuple[List[int], List[int]]:
    if len(input_ids) <= max_length:
        return input_ids, labels
    if truncation_side == 'left':
        return input_ids[-max_length:], labels[-max_length:]
    return input_ids[:max_length], labels[:max_length]


def _build_input_and_labels_from_messages(tokenizer, messages: Messages, *, max_length: int, truncation_side: str) -> Tuple[List[int], List[int]]:
    full_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    full_ids = _encode_text(tokenizer, full_text)

    labels = [-100] * len(full_ids)

    for i, m in enumerate(messages):
        if m.get('role') != 'assistant':
            continue
        content = m.get('content')
        if content is None:
            continue
        prefix_messages = messages[:i]
        prompt_text = _apply_chat_template(tokenizer, prefix_messages, add_generation_prompt=True)
        upto_text = _apply_chat_template(tokenizer, prefix_messages + [m], add_generation_prompt=False)
        prompt_ids = _encode_text(tokenizer, prompt_text)
        upto_ids = _encode_text(tokenizer, upto_text)

        common = 0
        max_common = min(len(prompt_ids), len(upto_ids))
        while common < max_common and prompt_ids[common] == upto_ids[common]:
            common += 1
        start = common
        end = len(upto_ids)
        if end > len(full_ids):
            continue
        if start >= end:
            continue
        for pos in range(start, end):
            labels[pos] = full_ids[pos]

    input_ids, labels = _truncate(full_ids, labels, max_length, truncation_side)
    return input_ids, labels


def _row_to_input_and_labels(tokenizer, row: Mapping[str, Any], *, max_length: int, truncation_side: str) -> Optional[Dict[str, Any]]:
    if 'messages' in row:
        messages = row['messages']
        if not isinstance(messages, list) or len(messages) == 0:
            return None
        input_ids, labels = _build_input_and_labels_from_messages(
            tokenizer, messages, max_length=max_length, truncation_side=truncation_side)
        if all(x == -100 for x in labels):
            return None
        return {'input_ids': input_ids, 'labels': labels}

    if 'prompt' in row and 'response' in row:
        prompt = str(row['prompt'])
        response = str(row['response'])
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = _build_input_and_labels_from_messages(
            tokenizer, messages, max_length=max_length, truncation_side=truncation_side)
        if all(x == -100 for x in labels):
            return None
        return {'input_ids': input_ids, 'labels': labels}

    if 'instruction' in row and 'output' in row:
        instruction = str(row['instruction'])
        input_text = str(row.get('input', ''))
        prompt = instruction if input_text == '' else f'{instruction}\n{input_text}'
        response = str(row['output'])
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = _build_input_and_labels_from_messages(
            tokenizer, messages, max_length=max_length, truncation_side=truncation_side)
        if all(x == -100 for x in labels):
            return None
        return {'input_ids': input_ids, 'labels': labels}

    if 'text' in row:
        text = row['text']
        if text is None:
            return None
        input_ids = _encode_text(tokenizer, str(text))
        input_ids, _ = _truncate(input_ids, input_ids, max_length, truncation_side)
        labels = input_ids[:]
        return {'input_ids': input_ids, 'labels': labels}

    return None


class SFTDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f['input_ids']) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = int(self.tokenizer.pad_token_id)
        for f in features:
            ids = list(f['input_ids'])
            lab = list(f['labels'])
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def _maybe_apply_lora(model, args: ScriptArgs):
    if args.lora_rank <= 0:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError('LoRA requested but peft is not available') from e
    target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    if not target_modules:
        raise ValueError('lora_target_modules is empty')
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias='none',
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    train_dataset, eval_dataset = _load_and_split_dataset(args.dataset, seed=args.seed, split_ratio=args.split_dataset_ratio)

    def map_fn(row: Dict[str, Any]) -> Dict[str, Any]:
        res = _row_to_input_and_labels(tokenizer, row, max_length=args.max_length, truncation_side=args.truncation_side)
        return {} if res is None else res

    train_dataset = train_dataset.map(map_fn, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.filter(lambda x: x.get('input_ids') is not None)
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(map_fn, remove_columns=eval_dataset.column_names)
        eval_dataset = eval_dataset.filter(lambda x: x.get('input_ids') is not None)
        if args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_resolve_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if hasattr(model, 'resize_token_embeddings') and len(tokenizer) > getattr(model.config, 'vocab_size', len(tokenizer)):
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    model = _maybe_apply_lora(model, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy='steps' if eval_dataset is not None else 'no',
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

