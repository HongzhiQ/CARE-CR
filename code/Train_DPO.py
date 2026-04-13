import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)


def _as_bool(v: str) -> bool:
    return str(v).lower() in {'1', 'true', 'yes', 'y', 't'}


def _resolve_dtype(name: str):
    if name == 'auto':
        return None
    return {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }[name]


def _ensure_last_assistant(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    if not messages:
        raise ValueError('messages is empty')
    last = messages[-1]
    if last.get('role') != 'assistant':
        raise ValueError('messages last role is not assistant')
    content = last.get('content')
    if content is None:
        raise ValueError('messages last content is None')
    prompt_messages = messages[:-1]
    return prompt_messages, content


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts = []
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        parts.append(f'[{role}]\n{content}')
    return '\n'.join(parts)


def _row_to_pair_text(row: Dict[str, Any], tokenizer) -> Tuple[str, str]:
    if 'messages' in row and 'rejected_response' in row:
        prompt_messages, chosen = _ensure_last_assistant(row['messages'])
        rejected = row['rejected_response']
        if rejected is None or rejected == chosen:
            raise ValueError('rejected_response is None or equals chosen')
        chosen_messages = prompt_messages + [{'role': 'assistant', 'content': chosen}]
        rejected_messages = prompt_messages + [{'role': 'assistant', 'content': rejected}]
        return _apply_chat_template(tokenizer, chosen_messages), _apply_chat_template(tokenizer, rejected_messages)

    if 'prompt' in row and 'chosen' in row and 'rejected' in row:
        prompt = row['prompt']
        chosen = row['chosen']
        rejected = row['rejected']
        chosen_text = f'{prompt}\n{chosen}'
        rejected_text = f'{prompt}\n{rejected}'
        return chosen_text, rejected_text

    raise ValueError('Unsupported row schema')


def _sequence_logps(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f'logits must be 3D, got shape={tuple(logits.shape)}')
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError('input_ids and attention_mask must be 2D')
    if logits.shape[0] != input_ids.shape[0] or logits.shape[1] != input_ids.shape[1]:
        raise ValueError('logits and input_ids shape mismatch')

    log_probs = F.log_softmax(logits, dim=-1)
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    token_logps = torch.gather(log_probs[:, :-1, :], dim=-1, index=next_token_ids).squeeze(-1)
    token_mask = attention_mask[:, 1:].to(dtype=token_logps.dtype)
    return (token_logps * token_mask).sum(dim=-1)


class DPODataCollator:

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_texts = [f['chosen_text'] for f in features]
        rejected_texts = [f['rejected_text'] for f in features]

        chosen_enc = self.tokenizer(
            chosen_texts,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        rejected_enc = self.tokenizer(
            rejected_texts,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        def to_features(enc: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
            n = len(enc['input_ids'])
            return [{k: enc[k][i] for k in enc.keys()} for i in range(n)]

        chosen_batch = self._collator(to_features(chosen_enc))
        rejected_batch = self._collator(to_features(rejected_enc))

        input_ids = torch.cat([chosen_batch['input_ids'], rejected_batch['input_ids']], dim=0)
        attention_mask = torch.cat([chosen_batch['attention_mask'], rejected_batch['attention_mask']], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


class DPOTrainerStandalone(Trainer):

    def __init__(self, *, ref_model: torch.nn.Module, beta: float, reference_free: bool, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = float(beta)
        self.reference_free = bool(reference_free)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size = input_ids.shape[0] // 2
        if batch_size == 0 or input_ids.shape[0] != 2 * batch_size:
            raise ValueError('Batch must contain chosen and rejected sequences concatenated on dim=0')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        policy_logps = _sequence_logps(outputs.logits, input_ids, attention_mask)
        policy_chosen, policy_rejected = torch.split(policy_logps, batch_size, dim=0)

        if self.reference_free:
            ref_chosen = policy_chosen.detach().new_zeros(policy_chosen.shape)
            ref_rejected = policy_rejected.detach().new_zeros(policy_rejected.shape)
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                ref_logps = _sequence_logps(ref_outputs.logits, input_ids, attention_mask)
                ref_chosen, ref_rejected = torch.split(ref_logps, batch_size, dim=0)

        pi_logratios = policy_chosen - policy_rejected
        ref_logratios = ref_chosen - ref_rejected

        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        if return_outputs:
            out = {
                'policy_chosen_logps': policy_chosen.detach(),
                'policy_rejected_logps': policy_rejected.detach(),
                'ref_chosen_logps': ref_chosen.detach(),
                'ref_rejected_logps': ref_rejected.detach(),
            }
            return loss, out
        return loss


def _maybe_apply_lora(model, args):
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
    model = get_peft_model(model, lora_config)
    return model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ref_model', type=str, default='')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--reference_free', type=str, default='false')

    parser.add_argument('--max_length', type=int, default=2048)
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
    parser.add_argument('--gradient_checkpointing', type=str, default='true')
    parser.add_argument('--torch_dtype', type=str, default='bfloat16', choices=['auto', 'float16', 'bfloat16', 'float32'])
    parser.add_argument('--bf16', type=str, default='true')
    parser.add_argument('--fp16', type=str, default='false')
    parser.add_argument('--trust_remote_code', type=str, default='true')
    parser.add_argument('--split_dataset_ratio', type=float, default=0.1)
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)

    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

    return parser.parse_args()


def main():
    args = _parse_args()
    set_seed(args.seed)

    trust_remote_code = _as_bool(args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    dataset = load_dataset('json', data_files={'data': args.dataset})['data']
    split_ratio = max(0.0, min(0.99, float(args.split_dataset_ratio)))
    if split_ratio > 0:
        split = dataset.train_test_split(test_size=split_ratio, seed=args.seed, shuffle=True)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = dataset
        eval_dataset = None

    def map_fn(row):
        chosen_text, rejected_text = _row_to_pair_text(row, tokenizer)
        return {'chosen_text': chosen_text, 'rejected_text': rejected_text}

    train_dataset = train_dataset.map(map_fn, remove_columns=train_dataset.column_names)
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(map_fn, remove_columns=eval_dataset.column_names)
        if args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))

    torch_dtype = _resolve_dtype(args.torch_dtype)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(policy_model.config, 'use_cache'):
        policy_model.config.use_cache = False
    if hasattr(policy_model, 'resize_token_embeddings') and len(tokenizer) > getattr(policy_model.config, 'vocab_size', len(tokenizer)):
        policy_model.resize_token_embeddings(len(tokenizer))

    ref_model_name = args.ref_model.strip() or args.model
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(ref_model.config, 'use_cache'):
        ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if _as_bool(args.gradient_checkpointing):
        policy_model.gradient_checkpointing_enable()
        if hasattr(policy_model, 'enable_input_require_grads'):
            policy_model.enable_input_require_grads()

    policy_model = _maybe_apply_lora(policy_model, args)

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
        bf16=_as_bool(args.bf16),
        fp16=_as_bool(args.fp16),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    collator = DPODataCollator(tokenizer=tokenizer, max_length=args.max_length)
    trainer = DPOTrainerStandalone(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        ref_model=ref_model,
        beta=args.beta,
        reference_free=_as_bool(args.reference_free),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'final_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

