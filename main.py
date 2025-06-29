#!/usr/bin/env python3
"""
FTRANS - Fast Transformer Acceleration with Block Circulant Matrices
A comprehensive project for accelerating transformers using block circulant matrix compression
and FPGA simulation with CPU/GPU comparison.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from src.model.transformer import create_bert_like_model
from src.compression.compression import compress_transformer
from src.fpga_sim.simulator import main as run_fpga_simulation
from src.fpga_sim.fpga_simulator import comprehensive_platform_comparison, print_comparison_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='FTRANS - Fast Transformer Acceleration with Block Circulant Matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --run-all

  # Run with custom parameters
  python main.py --run-all --seq-length 256 --batch-size 4 --block-size 8

  # Run only platform comparison
  python main.py --compare-platforms

  # Run only compression
  python main.py --compress-only

  # Run only FPGA simulation
  python main.py --simulate-only
        """
    )
    
    # Main operation modes
    parser.add_argument('--run-all', action='store_true', 
                        help='Run the complete pipeline: model creation, compression, and simulation')
    parser.add_argument('--compare-platforms', action='store_true',
                        help='Run comprehensive CPU/GPU/FPGA platform comparison')
    parser.add_argument('--compress-only', action='store_true',
                        help='Only run model compression')
    parser.add_argument('--simulate-only', action='store_true',
                        help='Only run FPGA simulation')
    
    # Model parameters
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Hidden dimension size (default: 512)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--vocab-size', type=int, default=30000,
                        help='Vocabulary size (default: 30000)')
    
    # Compression parameters
    parser.add_argument('--block-size', type=int, default=4,
                        help='Block size for circulant matrix compression (default: 4)')
    parser.add_argument('--compression-ratio', type=float, default=0.5,
                        help='Target compression ratio (default: 0.5)')
    
    # Simulation parameters
    parser.add_argument('--seq-length', type=int, default=128,
                        help='Sequence length for simulation (default: 128)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for simulation (default: 1)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for results (default: output)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def create_output_directories(base_dir):
    """Create necessary output directories."""
    directories = [
        base_dir,
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'plots')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories


def run_model_creation(args):
    """Create the original transformer model."""
    print("\n" + "="*60)
    print("STEP 1: CREATING TRANSFORMER MODEL")
    print("="*60)
    
    model_config = {
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'vocab_size': args.vocab_size,
        'max_seq_length': args.seq_length
    }
    
    print(f"Model configuration: {model_config}")
    
    start_time = time.time()
    model = create_bert_like_model(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_classes=2
    )
    creation_time = time.time() - start_time
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'models', 'original_model.pth')
    torch.save(model.state_dict(), model_path)
    
    print(f"✓ Model created successfully in {creation_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    return model, model_path


def run_compression(args, model, model_path):
    """Compress the model using block circulant matrices."""
    print("\n" + "="*60)
    print("STEP 2: COMPRESSING MODEL WITH BLOCK CIRCULANT MATRICES")
    print("="*60)
    
    print(f"Block size: {args.block_size}")
    print(f"Target compression ratio: {args.compression_ratio}")
    
    start_time = time.time()
    compressed_model, compression_rate = compress_transformer(
        model=model,
        block_size=args.block_size
    )
    compression_time = time.time() - start_time
    
    # Save the compressed model
    compressed_model_path = os.path.join(args.output_dir, 'models', 'compressed_model.pth')
    torch.save(compressed_model.state_dict(), compressed_model_path)
    
    print(f"✓ Model compressed successfully in {compression_time:.2f} seconds")
    print(f"✓ Compressed model saved to: {compressed_model_path}")
    
    # Print compression statistics
    original_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    actual_compression_ratio = compressed_params / original_params
    
    print(f"✓ Original parameters: {original_params:,}")
    print(f"✓ Compressed parameters: {compressed_params:,}")
    print(f"✓ Actual compression ratio: {actual_compression_ratio:.3f}")
    print(f"✓ Size reduction: {(1 - actual_compression_ratio) * 100:.1f}%")
    print(f"✓ Theoretical compression rate: {compression_rate:.2f}x")
    
    return compressed_model, compressed_model_path


def run_platform_comparison(args):
    """Run comprehensive platform comparison."""
    print("\n" + "="*60)
    print("COMPREHENSIVE PLATFORM COMPARISON")
    print("="*60)
    
    model_config = {
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'num_heads': args.num_heads,
        'seq_length': args.seq_length,
        'batch_size': args.batch_size
    }
    
    print(f"Model configuration: {model_config}")
    print(f"Block size for FPGA compression: {args.block_size}")
    
    start_time = time.time()
    results = comprehensive_platform_comparison(
        model_config=model_config,
        block_size=args.block_size
    )
    simulation_time = time.time() - start_time
    
    # Print results
    print_comparison_results(results)
    print(f"\n✓ Platform comparison completed in {simulation_time:.2f} seconds")
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, 'results', 'platform_comparison.json')
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    return results


def run_fpga_simulation(args, original_model_path=None, compressed_model_path=None):
    """Run FPGA simulation."""
    print("\n" + "="*60)
    print("STEP 3: RUNNING FPGA SIMULATION")
    print("="*60)
    
    # Prepare simulation arguments
    sim_args = [
        '--seq-length', str(args.seq_length),
        '--batch-size', str(args.batch_size),
        '--block-size', str(args.block_size),
        '--output-dir', os.path.join(args.output_dir, 'results'),
        '--num-layers', str(args.num_layers),
        '--hidden-size', str(args.hidden_size),
        '--num-heads', str(args.num_heads)
    ]
    
    if original_model_path:
        sim_args.extend(['--original-model', original_model_path])
    if compressed_model_path:
        sim_args.extend(['--compressed-model', compressed_model_path])
    
    if args.verbose:
        sim_args.append('--verbose')
    
    # Run simulation by calling the simulator directly
    import sys
    original_argv = sys.argv
    sys.argv = ['simulator.py'] + sim_args
    
    start_time = time.time()
    try:
        run_fpga_simulation()
        simulation_time = time.time() - start_time
        print(f"✓ FPGA simulation completed in {simulation_time:.2f} seconds")
    finally:
        sys.argv = original_argv


def main():
    """Main function to run the FTRANS pipeline."""
    args = parse_args()
    
    # Create output directories
    create_output_directories(args.output_dir)
    
    print("FTRANS - Fast Transformer Acceleration with Block Circulant Matrices")
    print("="*70)
    
    if args.compare_platforms:
        # Run comprehensive platform comparison
        run_platform_comparison(args)
        
    elif args.run_all:
        # Run complete pipeline
        print("Running complete FTRANS pipeline...")
        
        # Step 1: Create model
        model, model_path = run_model_creation(args)
        
        # Step 2: Compress model
        compressed_model, compressed_model_path = run_compression(args, model, model_path)
        
        # Step 3: Run FPGA simulation
        run_fpga_simulation(args, model_path, compressed_model_path)
        
        # Step 4: Run platform comparison
        print("\n" + "="*60)
        print("STEP 4: PLATFORM COMPARISON")
        print("="*60)
        run_platform_comparison(args)
        
    elif args.compress_only:
        # Only run compression
        print("Running model compression only...")
        model, model_path = run_model_creation(args)
        run_compression(args, model, model_path)
        
    elif args.simulate_only:
        # Only run simulation
        print("Running FPGA simulation only...")
        run_fpga_simulation(args)
        
    else:
        # Default: run complete pipeline
        print("No specific mode selected. Running complete pipeline...")
        args.run_all = True
        main()
        return
    
    print("\n" + "="*70)
    print("FTRANS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Results saved in: {args.output_dir}")
    
    # Print summary of what was generated
    print("\nGenerated files:")
    if os.path.exists(os.path.join(args.output_dir, 'models')):
        models_dir = os.path.join(args.output_dir, 'models')
        if os.path.exists(os.path.join(models_dir, 'original_model.pth')):
            print(f"  • Original model: {os.path.join(models_dir, 'original_model.pth')}")
        if os.path.exists(os.path.join(models_dir, 'compressed_model.pth')):
            print(f"  • Compressed model: {os.path.join(models_dir, 'compressed_model.pth')}")
    
    results_dir = os.path.join(args.output_dir, 'results')
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                print(f"  • Results: {os.path.join(results_dir, file)}")
    
    plots_dir = os.path.join(args.output_dir, 'plots')
    if os.path.exists(plots_dir):
        for file in os.listdir(plots_dir):
            if file.endswith('.png'):
                print(f"  • Plot: {os.path.join(plots_dir, file)}")


if __name__ == "__main__":
    main() 