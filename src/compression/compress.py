import torch
import sys
import os
import argparse
import time
from pathlib import Path
from transformers import AutoModelForSequenceClassification

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.compression.compression import compress_transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Compress a transformer model using BCM')
    parser.add_argument('--model-name', type=str, default='textattack/bert-base-uncased-imdb', help='Pretrained model name')
    parser.add_argument('--block-size', type=int, default=4, help='Block size for BCM compression')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading pretrained model {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    
    # Count parameters in the original model
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model has {original_params:,} parameters")
    
    # Compress the model
    print(f"Compressing model with block size {args.block_size}...")
    start_time = time.time()
    compressed_model, compression_rate = compress_transformer(model, block_size=args.block_size)
    compression_time = time.time() - start_time
    
    # Count parameters in the compressed model
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    print(f"Compression completed in {compression_time:.2f} seconds")
    print(f"Compressed model effectively has {original_params // compression_rate:,} parameters")
    print(f"Achieved compression rate: {compression_rate:.2f}x")
    
    # Save the models
    original_model_path = os.path.join(args.output_dir, 'original_model.pt')
    compressed_model_path = os.path.join(args.output_dir, 'compressed_model.pt')
    
    print(f"Saving original model to {original_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, original_model_path)
    
    print(f"Saving compressed model to {compressed_model_path}")
    torch.save({
        'model_state_dict': compressed_model.state_dict(),
        'compression_rate': compression_rate,
        'args': vars(args)
    }, compressed_model_path)
    
    print("Compression process completed successfully!")


if __name__ == "__main__":
    main() 