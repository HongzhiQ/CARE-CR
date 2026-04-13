import os
from typing import Optional

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PreferencePredictor(nn.Module):
    """Bert-based Preference Weight Prediction Model"""
    def __init__(self, model_name=None, local_model_path=None, num_labels=7, dropout=0.1):
        super(PreferencePredictor, self).__init__()
        
        loaded = False
        actual_model_path = None
        
        if local_model_path and os.path.exists(local_model_path):
            try:
                self.bert = AutoModel.from_pretrained(local_model_path, local_files_only=True)
                print(f"✓ Successfully loaded the model locally: {local_model_path}")
                loaded = True
                actual_model_path = local_model_path
            except Exception as e:
                print(f"✗ Failed to load the model locally: {str(e)[:100]}")
        
        if not loaded:
            model_candidates = []
            if model_name:
                model_candidates.append(model_name)
            model_candidates.extend([
                'mental/mental-bert-base-zh',
                'hfl/chinese-bert-wwm-ext',
                'bert-base-chinese',
            ])
            
            for candidate in model_candidates:
                try:
                    self.bert = AutoModel.from_pretrained(candidate)
                    print(f"✓ Successfully loaded the model from HuggingFace: {candidate}")
                    loaded = True
                    actual_model_path = candidate
                    break
                except Exception as e:
                    print(f"✗ Failed to load {candidate}: {str(e)[:100]}")
                    continue
        
        if not loaded:
            raise RuntimeError("Failed to load any BERT model. Please check the local path or network connection.")
        
        self.actual_model_path = actual_model_path
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        mlp_hidden = hidden_size // 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        probs = self.softmax(logits)
        return probs




def load_tokenizer(
    save_dir: Optional[str] = None,
    local_model_path: Optional[str] = None,
    model_name: Optional[str] = None,
):
    tokenizer = None
    if save_dir and os.path.isdir(save_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)
        except Exception:
            tokenizer = None

    if tokenizer is None and local_model_path and os.path.isdir(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        except Exception:
            tokenizer = None

    if tokenizer is None and model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        for candidate in ["mental/mental-bert-base-zh", "hfl/chinese-bert-wwm-ext", "bert-base-chinese"]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                break
            except Exception:
                continue

    if tokenizer is None:
        raise RuntimeError("Unable to load a tokenizer from the provided inputs or fallback candidates.")
    return tokenizer


def resolve_checkpoint(save_dir: Optional[str] = None, checkpoint_path: Optional[str] = None) -> Optional[str]:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    if save_dir and os.path.isdir(save_dir):
        best_path = os.path.join(save_dir, "best_model.pt")
        final_path = os.path.join(save_dir, "final_model.pt")
        if os.path.isfile(best_path):
            return best_path
        if os.path.isfile(final_path):
            return final_path
    return None