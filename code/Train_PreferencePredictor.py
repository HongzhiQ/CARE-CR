import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from layers.Bert_Predictor import PreferencePredictor
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dims
DIMENSIONS = ['同理心', '积极性', '理性', '可执行性', '具体性', '可读性']

class PreferenceDataset(Dataset):
    def __init__(self, texts, weights, tokenizer, max_length=512):
        self.texts = texts
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        weights = self.weights[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(weights),
            'text': text
        }

def load_data(csv_path):
    """Loading the training data"""
    print(f"Loading data: {csv_path}")
    csv_name = os.path.basename(csv_path).lower()
    texts = None
    weights = None
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df['x'].values
    weights = df[[c for c in df.columns if c in DIMENSIONS]].values.astype(float)
    sums = weights.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    weights = weights / sums
        
    return texts, weights

def kl_divergence_loss(pred, target, epsilon=1e-8):
    pred = torch.clamp(pred, min=epsilon)
    target = torch.clamp(target, min=epsilon)
    kl = target * torch.log(target / pred)
    return kl.sum(dim=1).mean()

def composite_loss(pred, target, alpha=0.3, epsilon=1e-8):
    kl = kl_divergence_loss(pred, target, epsilon)
    mse = torch.mean((pred - target) ** 2)
    return kl + alpha * mse

def plot_loss_curves(train_losses, val_losses, save_dir, loss_type='kl'):
    plt.figure(figsize=(8, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='train', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(f'Loss ({loss_type.upper()})', fontsize=12)
    plt.title('Train and validate loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_curve_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    

def test_model(model, test_dataset, test_loader, device, save_dir):
    model.eval()
    
    all_texts = []
    all_true_weights = []
    all_pred_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='test'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_texts.extend(texts)
            all_true_weights.append(labels.cpu().numpy())
            all_pred_weights.append(outputs.cpu().numpy())
    
    all_true_weights = np.concatenate(all_true_weights, axis=0)
    all_pred_weights = np.concatenate(all_pred_weights, axis=0)
    
    results_df = pd.DataFrame()
    results_df['x'] = all_texts
    
    for i, dim in enumerate(DIMENSIONS):
        results_df[f'True_{dim}'] = all_true_weights[:, i]
    
    for i, dim in enumerate(DIMENSIONS):
        results_df[f'Pred_{dim}'] = all_pred_weights[:, i]
    
    results_path = os.path.join(save_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    mse = mean_squared_error(all_true_weights, all_pred_weights)
    print(f"Vaild - MSE: {mse:.6f}")
    
    print("\n MSE of each dimension:")
    for i, dim in enumerate(DIMENSIONS):
        dim_mse = mean_squared_error(all_true_weights[:, i], all_pred_weights[:, i])
        print(f"  {dim}: {dim_mse:.6f}")
    
    return results_df

def train_model(
    model_name,
    local_model_path,
    csv_path,
    batch_size,
    learning_rate,
    num_epochs,
    max_length,
    loss_type,
    train_ratio,
    save_dir,
    alpha,
    freeze_bert,
    weight_decay,
    unfreeze_last_n_layers
):
    
    os.makedirs(save_dir, exist_ok=True)
    
    texts, weights = load_data(csv_path)
    
    train_texts, val_texts, train_weights, val_weights = train_test_split(
        texts, weights, test_size=1-train_ratio, random_state=42
    )

    tokenizer = None
    
    if local_model_path and os.path.exists(local_model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            print(f"✓ Successfully loaded the tokenizer locally: {local_model_path}")
        except Exception as e:
            print(f"✗ Failed to load the tokenizer locally: {str(e)[:100]}")
    
    if tokenizer is None:
        tokenizer_candidates = []
        if model_name:
            tokenizer_candidates.append(model_name)
        tokenizer_candidates.extend([
            'mental/mental-bert-base-zh',
            'hfl/chinese-bert-wwm-ext',
            'bert-base-chinese',
        ])
        
        for candidate in tokenizer_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                print(f"✓ Successfully loaded tokenizer from HuggingFace: {candidate}")
                break
            except Exception as e:
                print(f"✗ Failed to load tokenizer {candidate}: {str(e)[:100]}")
                continue
    
    if tokenizer is None:
        raise RuntimeError("Failed to load any tokenizer. Please check the local path or network connection.")
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        texts = [item['text'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'text': texts
        }
    
    train_dataset = PreferenceDataset(train_texts, train_weights, tokenizer, max_length)
    val_dataset = PreferenceDataset(val_texts, val_weights, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = PreferencePredictor(model_name=model_name, local_model_path=local_model_path, num_labels=len(DIMENSIONS))
    model.to(device)
    
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
        if unfreeze_last_n_layers > 0 and hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "layer"):
            encoder_layers = model.bert.encoder.layer
            for layer in encoder_layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfreeze BERT's final {unfreeze_last_n_layers} layers")
        else:
            print("The BERT encoder has been frozen")
    else:
        print("The BERT encoder is co-trained with the classification head")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if loss_type == 'kl':
                loss = kl_divergence_loss(outputs, labels)
            elif loss_type == 'mse':
                loss = nn.functional.mse_loss(outputs, labels)
            else:
                loss = composite_loss(outputs, labels, alpha=alpha)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}')
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if loss_type == 'kl':
                    loss = kl_divergence_loss(outputs, labels)
                elif loss_type == 'mse':
                    loss = nn.functional.mse_loss(outputs, labels)
                else:
                    loss = composite_loss(outputs, labels, alpha=alpha)
                
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
                val_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        mse = mean_squared_error(val_targets, val_predictions)
        

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'mse': mse,
            }, model_path)
    
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_model_path)
    
    tokenizer.save_pretrained(save_dir)
        
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = test_model(model, val_dataset, val_loader, device, save_dir)
    
    plot_loss_curves(train_losses, val_losses, save_dir, loss_type)
    
    return model, tokenizer, train_losses, val_losses

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Preference-weight training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_model_path", type=str, default=None, help="Local pretrained model directory (preferred if provided and exists).")
    parser.add_argument("--model_name", type=str,default=None, help="HuggingFace model identifier (used if local_model_path is unavailable).")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the training CSV.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length.")
    parser.add_argument("--loss_type", type=str,default=None, choices=["kl", "mse", "kl_mse"], help="Training objective" )
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio.")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=None, help="MSE weight in the composite objective.")
    parser.add_argument('--freeze_bert', action='store_true', default=None, help='Freeze the BERT encoder and optimize only the prediction head.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay coefficient for AdamW optimization.')
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=2, help='When the encoder is frozen, additionally unfreeze the last N encoder layers for partial fine-tuning.')

    args = parser.parse_args()
    
    model, tokenizer, train_losses, val_losses = train_model(
        model_name=args.model_name,
        local_model_path=args.local_model_path,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        loss_type=args.loss_type,
        train_ratio=args.train_ratio,
        save_dir=args.save_dir,
        alpha=args.alpha,
        freeze_bert=args.freeze_bert,
        weight_decay=args.weight_decay,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers
    )
    
