import torch
import sys
import os
import argparse
import time
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model.transformer import create_bert_like_model
from src.compression.compression import compress_transformer


def parse_args():
    parser = argparse.ArgumentParser(description='Compress a transformer model using BCM')
    parser.add_argument('--vocab-size', type=int, default=30000, help='Vocabulary size for the model')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size of the transformer')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--block-size', type=int, default=4, help='Block size for BCM compression')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating a transformer model with {args.num_layers} layers, {args.hidden_size} hidden size, {args.num_heads} attention heads")
    model = create_bert_like_model(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
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