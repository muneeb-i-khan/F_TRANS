import torch
import numpy as np
import time
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class FPGASpecs:
    """
    Specifications for the FPGA hardware.
    """
    # Maximum number of DSP units available
    max_dsps: int = 2520  # Typical for medium-sized FPGA
    
    # Maximum on-chip memory (in MB)
    max_bram: float = 36.0  # MB
    
    # Clock frequency in MHz
    clock_freq: float = 300.0  # MHz
    
    # Power efficiency (operations per watt)
    ops_per_watt: float = 20.0e9  # 20 GOPS/Watt
    
    # Power consumption (in watts)
    power_consumption: float = 15.0  # Watts
    
    # Precision (bits)
    precision: int = 16  # 16-bit fixed point
    
    # Memory bandwidth (GB/s)
    memory_bandwidth: float = 12.0  # GB/s


class MatrixMultUnit:
    """
    Simulate matrix multiplication unit on FPGA.
    """
    def __init__(self, specs):
        self.specs = specs
        # Each DSP can perform one MAC (multiply-accumulate) operation per cycle
        self.max_parallel_macs = specs.max_dsps
    
    def compute_cycles(self, M, N, K):
        """
        Compute the number of cycles needed for M x K * K x N matrix multiplication.
        
        Args:
            M, N, K: Matrix dimensions
        
        Returns:
            Number of cycles
        """
        total_macs = M * N * K
        cycles_per_batch = total_macs / min(self.max_parallel_macs, total_macs)
        
        # Add overhead for memory access
        memory_overhead = (M * K + K * N + M * N) * 4 / (self.specs.memory_bandwidth * 1e9 / self.specs.clock_freq * 1e6)
        
        return int(cycles_per_batch + memory_overhead)


class FFTUnit:
    """
    Simulate Fast Fourier Transform unit for block circulant matrix operations.
    """
    def __init__(self, specs):
        self.specs = specs
        # FFT typically uses a butterfly network with log2(N) stages
        # Each stage uses DSPs for complex multiplications
    
    def compute_cycles(self, size):
        """
        Compute cycles for FFT of given size.
        
        Args:
            size: Size of the FFT
            
        Returns:
            Number of cycles
        """
        # For N-point FFT, we need log2(N) stages with N/2 complex multiplications each
        log2_n = int(np.log2(size))
        total_complex_muls = size * log2_n / 2
        
        # Each complex multiplication uses 4 real multiplications and 2 additions
        total_ops = total_complex_muls * 6
        
        # Assuming we can parallelize operations
        parallelism = min(self.specs.max_dsps // 6, size // 2)
        
        return int(total_ops / parallelism)


class BlockCirculantMatrixProcessor:
    """
    Simulate Block Circulant Matrix operations on FPGA.
    """
    def __init__(self, specs, block_size):
        self.specs = specs
        self.block_size = block_size
        self.fft_unit = FFTUnit(specs)
        self.matmul_unit = MatrixMultUnit(specs)
    
    def compute_cycles_for_bcm_matmul(self, M, N, K):
        """
        Compute cycles for block circulant matrix multiplication.
        
        For block circulant matrices, we can use FFT to convert convolution
        to element-wise multiplication, which is more efficient.
        
        Args:
            M, N, K: Matrix dimensions
            
        Returns:
            Number of cycles
        """
        # Number of blocks in each dimension
        blocks_M = (M + self.block_size - 1) // self.block_size
        blocks_N = (N + self.block_size - 1) // self.block_size
        blocks_K = (K + self.block_size - 1) // self.block_size
        
        # For each block-block multiplication:
        # 1. Perform FFT on the first row of each block
        # 2. Element-wise multiplication in frequency domain
        # 3. Inverse FFT
        
        # FFT of all blocks
        fft_cycles = (blocks_M * blocks_K + blocks_K * blocks_N) * self.fft_unit.compute_cycles(self.block_size)
        
        # Element-wise multiplication in frequency domain
        elem_mul_cycles = blocks_M * blocks_N * blocks_K * self.block_size
        
        # Inverse FFT for result blocks
        ifft_cycles = blocks_M * blocks_N * self.fft_unit.compute_cycles(self.block_size)
        
        return fft_cycles + elem_mul_cycles + ifft_cycles


class FPGASimulator:
    """
    Simulate the execution of a compressed transformer model on FPGA.
    """
    def __init__(self, fpga_specs=None, block_size=4):
        self.specs = fpga_specs if fpga_specs is not None else FPGASpecs()
        self.block_size = block_size
        self.bcm_processor = BlockCirculantMatrixProcessor(self.specs, block_size)
        self.matrix_mult_unit = MatrixMultUnit(self.specs)
        
        # Track resource utilization
        self.dsp_utilization = 0
        self.bram_utilization = 0
        
        # Performance metrics
        self.total_cycles = 0
        self.total_power = 0
    
    def simulate_linear_layer(self, in_features, out_features, batch_size, is_compressed=False):
        """
        Simulate execution of a linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            batch_size: Batch size for inference
            is_compressed: Whether the layer uses block circulant compression
            
        Returns:
            Dictionary with simulation results
        """
        if is_compressed:
            # Block circulant matrix multiplication
            cycles = self.bcm_processor.compute_cycles_for_bcm_matmul(batch_size, out_features, in_features)
            
            # Compute effective DSP utilization
            effective_dsps = min(self.specs.max_dsps, (in_features * out_features) // (self.block_size))
            
            # Memory footprint is reduced by a factor of block_size
            memory_footprint = (in_features * out_features * 4) / self.block_size  # 4 bytes per float32
            
        else:
            # Standard matrix multiplication
            cycles = self.matrix_mult_unit.compute_cycles(batch_size, out_features, in_features)
            
            # Compute effective DSP utilization
            effective_dsps = min(self.specs.max_dsps, batch_size * out_features)
            
            # Full memory footprint
            memory_footprint = in_features * out_features * 4  # 4 bytes per float32
        
        # Update resource utilization
        self.dsp_utilization = max(self.dsp_utilization, effective_dsps)
        self.bram_utilization += memory_footprint / (1024 * 1024)  # Convert to MB
        
        # Add cycles to total
        self.total_cycles += cycles
        
        # Compute metrics
        latency_ms = cycles / self.specs.clock_freq / 1000  # ms
        energy_mj = latency_ms * self.specs.power_consumption  # mJ
        
        return {
            'cycles': cycles,
            'latency_ms': latency_ms,
            'energy_mj': energy_mj,
            'dsp_utilization': effective_dsps / self.specs.max_dsps,
            'memory_mb': memory_footprint / (1024 * 1024)
        }
    
    def simulate_attention(self, hidden_size, num_heads, seq_length, batch_size, is_compressed=False):
        """
        Simulate a self-attention layer.
        
        Args:
            hidden_size: Size of hidden dimension
            num_heads: Number of attention heads
            seq_length: Sequence length
            batch_size: Batch size
            is_compressed: Whether the layer is compressed
            
        Returns:
            Dictionary with simulation results
        """
        head_dim = hidden_size // num_heads
        
        # 1. Linear projections for Q, K, V
        q_proj = self.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length, is_compressed)
        k_proj = self.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length, is_compressed)
        v_proj = self.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length, is_compressed)
        
        # 2. Attention scores: (batch_size, num_heads, seq_length, seq_length)
        # Matrix multiplication: (batch_size * num_heads, seq_length, head_dim) @ (batch_size * num_heads, head_dim, seq_length)
        attn_scores = self.simulate_linear_layer(head_dim, seq_length, batch_size * num_heads * seq_length, False)
        
        # 3. Softmax - relatively lightweight computation
        softmax_cycles = batch_size * num_heads * seq_length * seq_length * 2  # Approximate cycles for softmax
        self.total_cycles += softmax_cycles
        
        # 4. Apply attention weights to values
        # Matrix multiplication: (batch_size * num_heads, seq_length, seq_length) @ (batch_size * num_heads, seq_length, head_dim)
        context = self.simulate_linear_layer(seq_length, head_dim, batch_size * num_heads * seq_length, False)
        
        # 5. Output projection
        out_proj = self.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length, is_compressed)
        
        # Combine results
        total_cycles = q_proj['cycles'] + k_proj['cycles'] + v_proj['cycles'] + attn_scores['cycles'] + softmax_cycles + context['cycles'] + out_proj['cycles']
        total_latency = total_cycles / self.specs.clock_freq / 1000  # ms
        total_energy = total_latency * self.specs.power_consumption  # mJ
        
        return {
            'cycles': total_cycles,
            'latency_ms': total_latency,
            'energy_mj': total_energy
        }
    
    def simulate_ffn(self, hidden_size, ffn_size, batch_size, seq_length, is_compressed=False):
        """
        Simulate a feed-forward network layer.
        
        Args:
            hidden_size: Hidden dimension
            ffn_size: FFN intermediate dimension
            batch_size: Batch size
            seq_length: Sequence length
            is_compressed: Whether the layer is compressed
            
        Returns:
            Dictionary with simulation results
        """
        # 1. First linear layer (expansion)
        linear1 = self.simulate_linear_layer(hidden_size, ffn_size, batch_size * seq_length, is_compressed)
        
        # 2. Activation function (GELU) - lightweight compared to matrix multiplications
        activation_cycles = batch_size * seq_length * ffn_size * 5  # Approximate cycles for GELU
        self.total_cycles += activation_cycles
        
        # 3. Second linear layer (projection)
        linear2 = self.simulate_linear_layer(ffn_size, hidden_size, batch_size * seq_length, is_compressed)
        
        # Combine results
        total_cycles = linear1['cycles'] + activation_cycles + linear2['cycles']
        total_latency = total_cycles / self.specs.clock_freq / 1000  # ms
        total_energy = total_latency * self.specs.power_consumption  # mJ
        
        return {
            'cycles': total_cycles,
            'latency_ms': total_latency,
            'energy_mj': total_energy
        }
    
    def simulate_transformer_layer(self, hidden_size, ffn_size, num_heads, seq_length, batch_size, is_compressed=False):
        """
        Simulate one transformer layer.
        
        Args:
            hidden_size: Hidden dimension
            ffn_size: Feed-forward network size
            num_heads: Number of attention heads
            seq_length: Sequence length
            batch_size: Batch size
            is_compressed: Whether the layer uses compression
            
        Returns:
            Dictionary with simulation results
        """
        # 1. Self-attention
        attn_results = self.simulate_attention(hidden_size, num_heads, seq_length, batch_size, is_compressed)
        
        # 2. Layer norm - relatively lightweight computation
        norm1_cycles = batch_size * seq_length * hidden_size * 3  # Approximate cycles for LayerNorm
        self.total_cycles += norm1_cycles
        
        # 3. Feed-forward network
        ffn_results = self.simulate_ffn(hidden_size, ffn_size, batch_size, seq_length, is_compressed)
        
        # 4. Layer norm - relatively lightweight computation
        norm2_cycles = batch_size * seq_length * hidden_size * 3  # Approximate cycles for LayerNorm
        self.total_cycles += norm2_cycles
        
        # Combine results
        total_cycles = attn_results['cycles'] + norm1_cycles + ffn_results['cycles'] + norm2_cycles
        total_latency = total_cycles / self.specs.clock_freq / 1000  # ms
        total_energy = total_latency * self.specs.power_consumption  # mJ
        
        return {
            'cycles': total_cycles,
            'latency_ms': total_latency,
            'energy_mj': total_energy
        }
    
    def simulate_transformer(self, num_layers, hidden_size, ffn_size, num_heads, seq_length, batch_size, is_compressed=False):
        """
        Simulate the full transformer model.
        
        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            ffn_size: Feed-forward network size
            num_heads: Number of attention heads
            seq_length: Sequence length
            batch_size: Batch size
            is_compressed: Whether the model uses compression
            
        Returns:
            Dictionary with simulation results and resource utilization
        """
        # Reset metrics for a new simulation
        self.total_cycles = 0
        self.dsp_utilization = 0
        self.bram_utilization = 0
        
        # Embedding layer
        vocab_size = 30000  # Typical for BERT-like models
        embedding_memory = vocab_size * hidden_size * 4  # 4 bytes per float32
        self.bram_utilization += embedding_memory / (1024 * 1024)  # Convert to MB
        
        # Embedding lookup is memory-bound, estimate cycles based on memory bandwidth
        embedding_cycles = batch_size * seq_length * hidden_size * 4 / (self.specs.memory_bandwidth * 1e9 / self.specs.clock_freq * 1e6)
        self.total_cycles += int(embedding_cycles)
        
        # Simulate each transformer layer
        layer_results = []
        for i in range(num_layers):
            layer_result = self.simulate_transformer_layer(
                hidden_size, ffn_size, num_heads, seq_length, batch_size, is_compressed
            )
            layer_results.append(layer_result)
        
        # Calculate total metrics
        total_cycles = self.total_cycles
        total_latency_ms = total_cycles / self.specs.clock_freq / 1000
        total_energy_mj = total_latency_ms * self.specs.power_consumption
        
        # Check if we exceed FPGA resources
        exceeds_dsps = self.dsp_utilization > self.specs.max_dsps
        exceeds_bram = self.bram_utilization > self.specs.max_bram
        
        return {
            'cycles': total_cycles,
            'latency_ms': total_latency_ms,
            'energy_mj': total_energy_mj,
            'throughput_tokens_per_sec': batch_size * seq_length / (total_latency_ms / 1000),
            'dsp_utilization': self.dsp_utilization / self.specs.max_dsps,
            'bram_utilization_mb': self.bram_utilization,
            'exceeds_resources': exceeds_dsps or exceeds_bram,
            'layer_results': layer_results
        }


def compare_models(original_config, compressed_config, fpga_specs=None, block_size=4):
    """
    Compare the performance of original and compressed transformer models on FPGA.
    
    Args:
        original_config: Dictionary with configuration for the original model
        compressed_config: Dictionary with configuration for the compressed model
        fpga_specs: Optional FPGASpecs instance
        block_size: Block size used for compression
        
    Returns:
        Dictionary with comparison results
    """
    simulator = FPGASimulator(fpga_specs, block_size)
    
    # Simulate the original model
    original_results = simulator.simulate_transformer(
        num_layers=original_config['num_layers'],
        hidden_size=original_config['hidden_size'],
        ffn_size=original_config['hidden_size'] * 4,  # Standard expansion factor
        num_heads=original_config['num_heads'],
        seq_length=original_config['seq_length'],
        batch_size=original_config['batch_size'],
        is_compressed=False
    )
    
    # Simulate the compressed model
    compressed_results = simulator.simulate_transformer(
        num_layers=compressed_config['num_layers'],
        hidden_size=compressed_config['hidden_size'],
        ffn_size=compressed_config['hidden_size'] * 4,  # Standard expansion factor
        num_heads=compressed_config['num_heads'],
        seq_length=compressed_config['seq_length'],
        batch_size=compressed_config['batch_size'],
        is_compressed=True
    )
    
    # Calculate improvements
    latency_improvement = original_results['latency_ms'] / compressed_results['latency_ms']
    energy_improvement = original_results['energy_mj'] / compressed_results['energy_mj']
    throughput_improvement = compressed_results['throughput_tokens_per_sec'] / original_results['throughput_tokens_per_sec']
    
    return {
        'original': original_results,
        'compressed': compressed_results,
        'latency_improvement': latency_improvement,
        'energy_improvement': energy_improvement,
        'throughput_improvement': throughput_improvement,
        'compression_rate': block_size,  # Theoretical compression rate with block circulant matrices
    } 