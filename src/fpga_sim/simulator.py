import torch
import sys
import os
import argparse
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .fpga_simulator import (
    FPGASpecs, CPUSpecs, GPUSpecs, 
    comprehensive_platform_comparison, 
    print_comparison_results,
    compare_models
)

# Add the src directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='FPGA Transformer Simulator with Platform Comparison')
    parser.add_argument('--original-model', type=str, help='Path to original model checkpoint')
    parser.add_argument('--compressed-model', type=str, help='Path to compressed model checkpoint')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--block-size', type=int, default=4, help='Block size for compression')
    parser.add_argument('--output-dir', type=str, default='output/results', help='Output directory')
    parser.add_argument('--compare-platforms', action='store_true', help='Compare CPU, GPU, and FPGA')
    
    # Model configuration
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    
    return parser.parse_args()


def load_model_config(model_path):
    """
    Load model configuration from checkpoint (placeholder implementation).
    In a real implementation, this would load the actual model configuration.
    """
    # This is a placeholder - in practice, you'd load from the actual model file
    return {
        'num_layers': 6,
        'hidden_size': 512,
        'num_heads': 8
    }


def save_results(results, output_path):
    """
    Save results to JSON file.
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
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
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def create_comparison_plots(results, output_dir):
    """
    Create visualization plots for platform comparison.
    """
    platforms = ['cpu', 'gpu', 'fpga_original', 'fpga_compressed']
    platform_names = ['CPU', 'GPU', 'FPGA (Original)', 'FPGA (Compressed)']
    
    # Extract data for plotting
    latencies = []
    energies = []
    throughputs = []
    
    for platform in platforms:
        if platform in results:
            latencies.append(results[platform]['latency_ms'])
            energies.append(results[platform]['energy_mj'])
            throughputs.append(results[platform]['throughput_tokens_per_sec'])
        else:
            latencies.append(0)
            energies.append(0)
            throughputs.append(0)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Latency comparison
    bars1 = ax1.bar(platform_names, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Latency Comparison Across Platforms')
    ax1.set_yscale('log')
    for i, v in enumerate(latencies):
        if v > 0:
            ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Energy comparison
    bars2 = ax2.bar(platform_names, energies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Energy (mJ)')
    ax2.set_title('Energy Consumption Comparison')
    ax2.set_yscale('log')
    for i, v in enumerate(energies):
        if v > 0:
            ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # Throughput comparison
    bars3 = ax3.bar(platform_names, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_ylabel('Throughput (tokens/sec)')
    ax3.set_title('Throughput Comparison')
    for i, v in enumerate(throughputs):
        if v > 0:
            ax3.text(i, v, f'{v:.0f}', ha='center', va='bottom')
    
    # Speedup comparison (relative to CPU)
    if 'cpu' in results:
        cpu_latency = results['cpu']['latency_ms']
        speedups = [cpu_latency / lat if lat > 0 else 0 for lat in latencies]
        bars4 = ax4.bar(platform_names, speedups, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_ylabel('Speedup vs CPU')
        ax4.set_title('Latency Speedup Relative to CPU')
        for i, v in enumerate(speedups):
            if v > 0:
                ax4.text(i, v, f'{v:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'platform_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create energy efficiency plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Energy per token
    energy_per_token = []
    for platform in platforms:
        if platform in results:
            energy_per_token.append(results[platform]['energy_per_token'])
        else:
            energy_per_token.append(0)
    
    bars = ax.bar(platform_names, energy_per_token, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Energy per Token (mJ/token)')
    ax.set_title('Energy Efficiency Comparison')
    ax.set_yscale('log')
    
    for i, v in enumerate(energy_per_token):
        if v > 0:
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare_platforms:
        # Comprehensive platform comparison
        print("Running comprehensive platform comparison...")
        
        model_config = {
            'num_layers': args.num_layers,
            'hidden_size': args.hidden_size,
            'num_heads': args.num_heads,
            'seq_length': args.seq_length,
            'batch_size': args.batch_size
        }
        
        # Create hardware specifications
        cpu_specs = CPUSpecs()
        gpu_specs = GPUSpecs()
        fpga_specs = FPGASpecs()
        
        print(f"Model Configuration: {model_config}")
        print(f"Block size for FPGA compression: {args.block_size}")
        
        start_time = time.time()
        results = comprehensive_platform_comparison(
            model_config=model_config,
            cpu_specs=cpu_specs,
            gpu_specs=gpu_specs,
            fpga_specs=fpga_specs,
            block_size=args.block_size
        )
        simulation_time = time.time() - start_time
        
        # Print results
        print_comparison_results(results)
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
        
        # Save results
        results_file = os.path.join(args.output_dir, 'comprehensive_platform_comparison.json')
        save_results(results, results_file)
        print(f"Results saved to {results_file}")
        
        # Create plots
        create_comparison_plots(results, args.output_dir)
        print(f"Plots saved to {args.output_dir}")
        
    else:
        # Original FPGA-only comparison
        print("Running FPGA-only comparison...")
        
        # Load model configurations
        if args.original_model:
            original_config = load_model_config(args.original_model)
        else:
            original_config = {
                'num_layers': args.num_layers,
                'hidden_size': args.hidden_size,
                'num_heads': args.num_heads
            }
        
        if args.compressed_model:
            compressed_config = load_model_config(args.compressed_model)
        else:
            compressed_config = original_config.copy()
        
        # Add sequence length and batch size to configurations
        original_config['seq_length'] = args.seq_length
        original_config['batch_size'] = args.batch_size
        compressed_config['seq_length'] = args.seq_length
        compressed_config['batch_size'] = args.batch_size
        
        # Create FPGA specifications
        fpga_specs = FPGASpecs()
        
        # Run simulation
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
        print("\nFPGA Simulation Results:")
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
        
        # Save results
        results_file = os.path.join(args.output_dir, 'fpga_simulation_results.json')
        save_results(results, results_file)
        print(f"Results saved to {results_file}")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()