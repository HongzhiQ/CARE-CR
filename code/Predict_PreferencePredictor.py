import os
import argparse
import torch
from layers.Bert_Predictor import PreferencePredictor, load_tokenizer, resolve_checkpoint

# dims
DIMENSIONS = ['同理心', '积极性', '理性', '可执行性', '具体性', '可读性']



def predict(text, local_model_path=None, model_name=None, save_dir=None, checkpoint_path=None, max_length=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = load_tokenizer(save_dir, local_model_path, model_name)
    model = PreferencePredictor(model_name=model_name, local_model_path=local_model_path, num_labels=len(DIMENSIONS))
    model.to(device)
    ckpt = resolve_checkpoint(save_dir, checkpoint_path)
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model_state_dict'])

    model.eval()
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        probs = model(input_ids=input_ids, attention_mask=attention_mask)
    scores = probs.squeeze(0).cpu().numpy().tolist()
    return scores

def main():
    parser = argparse.ArgumentParser(description='Preference-weight prediction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--text', type=str, required=True, help='Input text to be scored.')
    parser.add_argument('--local_model_path', type=str, default=None, help='Absolute path to a local pretrained backbone (optional).')
    parser.add_argument('--model_name', type=str, default=None, help='HuggingFace model identifier for the backbone (optional).')
    parser.add_argument('--save_dir', type=str, default=None, help='Training output directory containing the tokenizer and checkpoints (optional).')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a checkpoint file to load (optional).')
    parser.add_argument('--max_length', type=int, default=None, help='Maximum sequence length used for tokenization (optional).')
    args = parser.parse_args()
    scores = predict(
        text=args.text,
        local_model_path=args.local_model_path,
        model_name=args.model_name,
        save_dir=args.save_dir,
        checkpoint_path=args.checkpoint_path,
        max_length=args.max_length,
    )
    for dim, s in zip(DIMENSIONS, scores):
        print(f'{dim}: {s:.6f}')

if __name__ == '__main__':
    main()
