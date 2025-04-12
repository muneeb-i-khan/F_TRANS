import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse
import time
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
    parser = argparse.ArgumentParser(description='Train a transformer model on the IMDB dataset')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--block-size', type=int, default=4, help='Block size for BCM compression')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max-seq-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--use-compressed', action='store_true', help='Use compressed model')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_and_prepare_dataset(tokenizer, dataset, max_seq_length):
    """
    Tokenize text and prepare dataset for training.
    """
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt'
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Convert to PyTorch tensors
    train_dataset = tokenized_dataset['train']
    test_dataset = tokenized_dataset['test']
    
    train_inputs = torch.stack([train_dataset['input_ids'][i] for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset['label'][i] for i in range(len(train_dataset))])
    
    test_inputs = torch.stack([test_dataset['input_ids'][i] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset['label'][i] for i in range(len(test_dataset))])
    
    return (
        TensorDataset(train_inputs, train_labels),
        TensorDataset(test_inputs, test_labels)
    )


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / len(progress_bar), 
            'acc': 100 * correct / total
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Initialize tokenizer and model
    print(f"Loading pretrained model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    
    # Prepare dataset
    print("Tokenizing and preparing dataset...")
    train_dataset, test_dataset = tokenize_and_prepare_dataset(
        tokenizer, dataset, args.max_seq_length
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Apply compression if requested
    if args.use_compressed:
        print(f"Compressing model with block size {args.block_size}...")
        model, compression_rate = compress_transformer(model, block_size=args.block_size)
        print(f"Achieved compression rate: {compression_rate:.2f}x")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    best_accuracy = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Evaluate
        eval_metrics = evaluate(model, test_dataloader, criterion, device)
        print(f"Evaluation - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f}, F1: {eval_metrics['f1']:.4f}")
        
        # Save the best model
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            model_prefix = 'compressed' if args.use_compressed else 'original'
            model_path = os.path.join(args.output_dir, f'{model_prefix}_best_model.pt')
            
            print(f"Saving best model to {model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': eval_metrics,
                'args': vars(args)
            }, model_path)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    # Final evaluation
    print("\nEvaluating final model...")
    final_metrics = evaluate(model, test_dataloader, criterion, device)
    print(f"Final Evaluation - Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}, F1: {final_metrics['f1']:.4f}")
    
    # Save final model
    model_prefix = 'compressed' if args.use_compressed else 'original'
    final_model_path = os.path.join(args.output_dir, f'{model_prefix}_final_model.pt')
    
    print(f"Saving final model to {final_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': final_metrics,
        'args': vars(args),
        'training_time': training_time
    }, final_model_path)
    
    print("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main() 