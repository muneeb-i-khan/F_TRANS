import torch
import sys
import os
import argparse
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.fpga_sim.fpga_simulator import FPGASpecs, FPGASimulator, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate transformer models on FPGA')
    parser.add_argument('--original-model', type=str, help='Path to the original model checkpoint')
    parser.add_argument('--compressed-model', type=str, help='Path to the compressed model checkpoint')
    parser.add_argument('--block-size', type=int, default=4, help='Block size used for compression')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length for inference')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    
    return parser.parse_args()


def load_model_config(model_path):
    """
    Load model configuration from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Dictionary with model configuration
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint.get('args', {})
    
    return {
        'num_layers': args.get('num_layers', 6),
        'hidden_size': args.get('hidden_size', 512),
        'num_heads': args.get('num_heads', 8)
    }


def plot_results(results, output_path):
    """
    Plot simulation results.
    
    Args:
        results: Dictionary with simulation results
        output_path: Path to save the plot
    """
    # Extract metrics
    orig_latency = results['original']['latency_ms']
    comp_latency = results['compressed']['latency_ms']
    
    orig_energy = results['original']['energy_mj']
    comp_energy = results['compressed']['energy_mj']
    
    orig_throughput = results['original']['throughput_tokens_per_sec']
    comp_throughput = results['compressed']['throughput_tokens_per_sec']
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Latency comparison
    axs[0].bar(['Original', 'Compressed'], [orig_latency, comp_latency])
    axs[0].set_title('Latency (ms)')
    axs[0].set_ylabel('Milliseconds')
    for i, v in enumerate([orig_latency, comp_latency]):
        axs[0].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Energy comparison
    axs[1].bar(['Original', 'Compressed'], [orig_energy, comp_energy])
    axs[1].set_title('Energy Consumption (mJ)')
    axs[1].set_ylabel('Millijoules')
    for i, v in enumerate([orig_energy, comp_energy]):
        axs[1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Throughput comparison
    axs[2].bar(['Original', 'Compressed'], [orig_throughput, comp_throughput])
    axs[2].set_title('Throughput (tokens/sec)')
    axs[2].set_ylabel('Tokens per second')
    for i, v in enumerate([orig_throughput, comp_throughput]):
        axs[2].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configurations
    print(f"Loading model configurations from checkpoints...")
    if args.original_model:
        original_config = load_model_config(args.original_model)
    else:
        # Use default configuration for original model
        original_config = {
            'num_layers': 6,
            'hidden_size': 512,
            'num_heads': 8
        }
    
    if args.compressed_model:
        compressed_config = load_model_config(args.compressed_model)
    else:
        # Use the same configuration as the original model for compressed model
        compressed_config = original_config.copy()
    
    # Add sequence length and batch size to configurations
    original_config['seq_length'] = args.seq_length
    original_config['batch_size'] = args.batch_size
    compressed_config['seq_length'] = args.seq_length
    compressed_config['batch_size'] = args.batch_size
    
    # Create FPGA specifications
    fpga_specs = FPGASpecs(
        max_dsps=2520,          # Typical for medium-sized FPGA
        max_bram=36.0,          # MB
        clock_freq=300.0,       # MHz
        power_consumption=15.0, # Watts
        memory_bandwidth=12.0   # GB/s
    )
    
    # Run simulation
    print(f"Running FPGA simulation...")
    print(f"Original model: {original_config}")
    print(f"Compressed model: {compressed_config}")
    
    start_time = time.time()
    results = compare_models(
        original_config=original_config,
        compressed_config=compressed_config,
        fpga_specs=fpga_specs,
        block_size=args.block_size
    )
    simulation_time = time.time() - start_time
    
    # Print results
    print("\nSimulation Results:")
    print(f"Simulation completed in {simulation_time:.2f} seconds")
    print(f"\nOriginal Model:")
    print(f"  Latency: {results['original']['latency_ms']:.2f} ms")
    print(f"  Energy: {results['original']['energy_mj']:.2f} mJ")
    print(f"  Throughput: {results['original']['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  DSP Utilization: {results['original']['dsp_utilization']*100:.2f}%")
    print(f"  BRAM Utilization: {results['original']['bram_utilization_mb']:.2f} MB")
    
    print(f"\nCompressed Model:")
    print(f"  Latency: {results['compressed']['latency_ms']:.2f} ms")
    print(f"  Energy: {results['compressed']['energy_mj']:.2f} mJ")
    print(f"  Throughput: {results['compressed']['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  DSP Utilization: {results['compressed']['dsp_utilization']*100:.2f}%")
    print(f"  BRAM Utilization: {results['compressed']['bram_utilization_mb']:.2f} MB")
    
    print(f"\nImprovements:")
    print(f"  Latency: {results['latency_improvement']:.2f}x")
    print(f"  Energy: {results['energy_improvement']:.2f}x")
    print(f"  Throughput: {results['throughput_improvement']:.2f}x")
    print(f"  Compression Rate: {results['compression_rate']}x")
    
    # Save results to file
    results_path = os.path.join(args.output_dir, 'fpga_simulation_results.json')
    with open(results_path, 'w') as f:
        # Convert some data types to make it JSON serializable
        serializable_results = {
            'original': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                         for k, v in results['original'].items() if k != 'layer_results'},
            'compressed': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                           for k, v in results['compressed'].items() if k != 'layer_results'},
            'latency_improvement': float(results['latency_improvement']),
            'energy_improvement': float(results['energy_improvement']),
            'throughput_improvement': float(results['throughput_improvement']),
            'compression_rate': results['compression_rate']
        }
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, 'fpga_simulation_results.png')
    plot_results(results, plot_path)
    print(f"Plots saved to {plot_path}")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()