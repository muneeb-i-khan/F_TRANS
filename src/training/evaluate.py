import torch
import torch.nn as nn
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.compression.compression import compress_transformer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a compressed transformer model on the IMDB dataset')
    parser.add_argument('--model-path', type=str, default='output/models/compressed_model.pt', help='Path to the compressed model')
    parser.add_argument('--model-name', type=str, default='textattack/bert-base-uncased-imdb', help='Pretrained model name for tokenizer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of test samples to evaluate (useful for quick evaluation)')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_and_prepare_dataset(tokenizer, dataset, max_seq_length):
    """
    Tokenize text and prepare dataset for evaluation.
    """
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_dataset


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            # If the model has more than 2 labels due to padding, consider only the first 2
            if logits.size(1) > 2:
                logits = logits[:, :2]

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load IMDB dataset
    print("Loading and preparing IMDB test dataset...")
    dataset = load_dataset("imdb", split='test')
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    test_dataset = tokenize_and_prepare_dataset(
        tokenizer, dataset, args.max_seq_length
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Load model
    print(f"Loading compressed model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
        
    # Load saved object
    saved_model_object = torch.load(args.model_path, map_location=device)
    
    # Determine state dict
    if isinstance(saved_model_object, dict) and 'model_state_dict' in saved_model_object:
        state_dict = saved_model_object['model_state_dict']
    elif isinstance(saved_model_object, dict):
        state_dict = saved_model_object
    else:
        # If a full model was saved, use its state_dict directly
        state_dict = saved_model_object.state_dict()

    # Initialize model with 2 labels (binary)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    
    # Adjust classifier weights/bias in the state dict if dimensions don't match
    state_classifier_weight = state_dict.get('classifier.weight')
    state_classifier_bias = state_dict.get('classifier.bias')
    if state_classifier_weight is not None and state_classifier_weight.shape[0] != model.classifier.weight.shape[0]:
        # Trim or pad weight rows to match
        desired_dim = model.classifier.weight.shape[0]
        if state_classifier_weight.shape[0] > desired_dim:
            state_dict['classifier.weight'] = state_classifier_weight[:desired_dim]
        else:
            pad_size = desired_dim - state_classifier_weight.shape[0]
            padding = torch.zeros(pad_size, state_classifier_weight.shape[1])
            state_dict['classifier.weight'] = torch.cat([state_classifier_weight, padding], dim=0)
    if state_classifier_bias is not None and state_classifier_bias.shape[0] != model.classifier.bias.shape[0]:
        desired_dim = model.classifier.bias.shape[0]
        if state_classifier_bias.shape[0] > desired_dim:
            state_dict['classifier.bias'] = state_classifier_bias[:desired_dim]
        else:
            pad_size = desired_dim - state_classifier_bias.shape[0]
            padding = torch.zeros(pad_size)
            state_dict['classifier.bias'] = torch.cat([state_classifier_bias, padding], dim=0)
    
    # Load state dict (non-strict to allow extra keys removed)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # Evaluate the model
    print("Evaluating compressed model...")
    eval_metrics = evaluate(model, test_dataloader, device)
    
    print("\nEvaluation Metrics for Compressed Model:")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"  F1-Score: {eval_metrics['f1']:.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 