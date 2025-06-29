#!/usr/bin/env python
import os
import argparse
import sys
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run the FTRANS pipeline')
    parser.add_argument('--compress', action='store_true', help='Compress the model')
    parser.add_argument('--simulate', action='store_true', help='Run FPGA simulation')
    parser.add_argument('--run-all', action='store_true', help='Run the entire pipeline')
    
    # Model parameters
    parser.add_argument('--model-name', type=str, default='textattack/bert-base-uncased-imdb', help='Pretrained model name')
    parser.add_argument('--block-size', type=int, default=4, help='Block size for BCM compression')
    
    # Simulation parameters
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length for simulation')
    parser.add_argument('--sim-batch-size', type=int, default=1, help='Batch size for simulation')
    
    # Output directories
    parser.add_argument('--output-dir', type=str, default='output', help='Main output directory')
    
    return parser.parse_args()


def setup_directories(args):
    """
    Create the necessary output directories.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    return model_dir, results_dir


def run_compression(args, model_dir):
    """
    Run the model compression step.
    """
    print("\n=== Running Model Compression ===")
    
    cmd = [
        sys.executable, 'src/compression/compress.py',
        '--model-name', args.model_name,
        '--block-size', str(args.block_size),
        '--output-dir', model_dir,
    ]
    
    subprocess.run(cmd, check=True)
    
    return (
        os.path.join(model_dir, 'original_model.pt'),
        os.path.join(model_dir, 'compressed_model.pt')
    )


def run_simulation(args, results_dir, original_model_path, compressed_model_path):
    """
    Run the FPGA simulation step.
    """
    print("\n=== Running FPGA Simulation ===")
    
    cmd = [
        sys.executable, 'src/fpga_sim/simulator.py',
        '--original-model', original_model_path,
        '--compressed-model', compressed_model_path,
        '--seq-length', str(args.seq_length),
        '--batch-size', str(args.sim_batch_size),
        '--output-dir', results_dir,
    ]
    
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    
    # Set up directories
    model_dir, results_dir = setup_directories(args)
    
    # Run the requested steps or all steps
    if args.run_all or args.compress:
        original_model_path, compressed_model_path = run_compression(args, model_dir)
    
    if args.run_all or args.simulate:
        # Check if model paths exist, otherwise use default names
        if not (args.run_all or args.compress):
            original_model_path = os.path.join(model_dir, 'original_model.pt')
            compressed_model_path = os.path.join(model_dir, 'compressed_model.pt')
        
        run_simulation(args, results_dir, original_model_path, compressed_model_path)
    
    print("\n=== FTRANS Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main() 