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


@dataclass
class CPUSpecs:
    """
    Specifications for CPU hardware.
    """
    clock_freq: float = 3000.0  # MHz (3 GHz)
    cores: int = 8
    power_consumption: float = 65.0  # Watts (TDP under load)
    idle_power: float = 15.0  # Watts (idle power consumption)
    memory_bandwidth: float = 50.0  # GB/s
    precision: int = 32  # 32-bit float


@dataclass
class GPUSpecs:
    """
    Specifications for GPU hardware.
    """
    cuda_cores: int = 2048  # CUDA cores
    clock_freq: float = 1500.0  # MHz
    power_consumption: float = 250.0  # Watts (peak compute power)
    idle_power: float = 75.0  # Watts (idle/base power consumption)
    memory_bandwidth: float = 500.0  # GB/s
    precision: int = 32  # 32-bit float
    memory_transfer_overhead: float = 0.001  # 1ms for PCIe transfers
    min_kernel_time: float = 0.0001  # 0.1ms minimum kernel execution time


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
        if size <= 1:
            return 1
            
        # Optimized FFT implementation for small sizes common in BCM
        # For N-point FFT, we need log2(N) stages
        log2_n = int(np.log2(size))
        
        # Optimized butterfly network with pipelining
        # Each stage can be processed in parallel with high efficiency
        if size <= 8:  # Small FFTs are very efficient on FPGA
            return log2_n * 2  # Highly optimized for small sizes
        else:
            # For larger FFTs, use more realistic cycle count
            total_complex_muls = size * log2_n / 2
            # Better parallelization for larger FFTs
            parallelism = min(self.specs.max_dsps // 4, size)
            return int(total_complex_muls * 4 / parallelism)


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
        
        # Optimized BCM multiplication with pipelining
        # 1. FFT operations can be pipelined and reused
        # 2. Element-wise multiplication is very efficient
        # 3. IFFT can also be pipelined
        
        # FFT cycles (with optimization for reuse and pipelining)
        unique_fft_ops = blocks_M * blocks_K + blocks_K * blocks_N
        fft_cycles = unique_fft_ops * self.fft_unit.compute_cycles(self.block_size)
        
        # Element-wise multiplication in frequency domain (very efficient)
        elem_mul_cycles = blocks_M * blocks_N * blocks_K * self.block_size // 4  # Parallel processing
        
        # Inverse FFT for result blocks (pipelined with computation)
        ifft_cycles = blocks_M * blocks_N * self.fft_unit.compute_cycles(self.block_size) // 2  # Pipelined
        
        # Total with pipelining efficiency (30% reduction due to overlapping operations)
        total_cycles = int((fft_cycles + elem_mul_cycles + ifft_cycles) * 0.7)
        
        return total_cycles


class CPUSimulator:
    """
    Simulate CPU execution of transformer operations.
    """
    def __init__(self, cpu_specs=None):
        self.specs = cpu_specs if cpu_specs is not None else CPUSpecs()
    
    def simulate_linear_layer(self, in_features, out_features, batch_size):
        """
        Simulate CPU execution of a linear layer.
        """
        # CPU uses optimized BLAS libraries (like MKL)
        total_ops = batch_size * in_features * out_features * 2  # MAC operations
        
        # CPU can achieve good parallelization with BLAS
        effective_throughput = self.specs.cores * self.specs.clock_freq * 1e6 * 2  # 2 ops per cycle per core
        
        # Memory bandwidth limitation
        memory_ops = (batch_size * in_features + in_features * out_features + batch_size * out_features) * 4
        memory_time = memory_ops / (self.specs.memory_bandwidth * 1e9)
        
        compute_time = total_ops / effective_throughput
        execution_time = max(compute_time, memory_time)  # Bottleneck
        
        # More realistic power consumption based on utilization
        if execution_time == compute_time:  # CPU-bound
            avg_power = self.specs.power_consumption  # Full TDP
        else:  # Memory-bound
            avg_power = self.specs.idle_power + 0.6 * (self.specs.power_consumption - self.specs.idle_power)
        
        energy = execution_time * avg_power
        
        return {
            'time_ms': execution_time * 1000,
            'energy_mj': energy * 1000,
            'throughput_ops_per_sec': total_ops / execution_time
        }


class GPUSimulator:
    """
    Simulate GPU execution of transformer operations.
    """
    def __init__(self, gpu_specs=None):
        self.specs = gpu_specs if gpu_specs is not None else GPUSpecs()
    
    def simulate_linear_layer(self, in_features, out_features, batch_size):
        """
        Simulate GPU execution of a linear layer.
        """
        # GPU excels at matrix multiplication
        total_ops = batch_size * in_features * out_features * 2  # MAC operations
        
        # GPU can achieve very high throughput for large matrices
        if total_ops > 1000000:  # Large matrix - GPU efficient
            effective_throughput = self.specs.cuda_cores * self.specs.clock_freq * 1e6 * 0.8  # 80% efficiency
            compute_power = self.specs.power_consumption  # Full power for large operations
        else:  # Small matrix - GPU less efficient due to launch overhead
            effective_throughput = self.specs.cuda_cores * self.specs.clock_freq * 1e6 * 0.3  # 30% efficiency
            compute_power = self.specs.power_consumption * 0.6  # Partial power utilization
        
        # Memory bandwidth (GPU has very high bandwidth)
        memory_ops = (batch_size * in_features + in_features * out_features + batch_size * out_features) * 4
        memory_time = memory_ops / (self.specs.memory_bandwidth * 1e9)
        
        compute_time = total_ops / effective_throughput
        
        # Add realistic GPU overheads
        kernel_launch_overhead = 0.00005  # 50 microseconds kernel launch
        memory_transfer_time = self.specs.memory_transfer_overhead  # PCIe transfer overhead
        
        # Minimum execution time (GPU kernels have setup costs)
        pure_compute_time = max(compute_time, memory_time)
        actual_compute_time = max(pure_compute_time, self.specs.min_kernel_time)
        
        # Total execution time includes all overheads
        total_execution_time = actual_compute_time + kernel_launch_overhead + memory_transfer_time
        
        # Energy calculation with more realistic power modeling
        # During computation: use compute_power
        # During overhead periods: use idle power
        compute_energy = actual_compute_time * compute_power
        overhead_energy = (kernel_launch_overhead + memory_transfer_time) * self.specs.idle_power
        
        total_energy = compute_energy + overhead_energy
        
        return {
            'time_ms': total_execution_time * 1000,
            'energy_mj': total_energy * 1000,
            'throughput_ops_per_sec': total_ops / total_execution_time
        }


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


def comprehensive_platform_comparison(model_config, cpu_specs=None, gpu_specs=None, fpga_specs=None, block_size=4):
    """
    Compare transformer execution across CPU, GPU, and FPGA platforms.
    
    Args:
        model_config: Dictionary with model configuration
        cpu_specs: Optional CPUSpecs instance
        gpu_specs: Optional GPUSpecs instance
        fpga_specs: Optional FPGASpecs instance
        block_size: Block size for FPGA compression
        
    Returns:
        Dictionary with comprehensive comparison results
    """
    # Initialize simulators
    cpu_sim = CPUSimulator(cpu_specs)
    gpu_sim = GPUSimulator(gpu_specs)
    fpga_sim = FPGASimulator(fpga_specs, block_size)
    
    # Extract model parameters
    num_layers = model_config['num_layers']
    hidden_size = model_config['hidden_size']
    ffn_size = hidden_size * 4
    num_heads = model_config['num_heads']
    seq_length = model_config['seq_length']
    batch_size = model_config['batch_size']
    
    results = {}
    
    # Minimum inference overhead (system-level costs)
    min_inference_time_ms = 0.5  # 0.5ms minimum for any inference
    
    # CPU Performance (Original Model Only - no compression)
    print("Simulating CPU performance...")
    cpu_total_time = 0
    cpu_total_energy = 0
    
    # Simulate each layer type on CPU
    for layer in range(num_layers):
        # Attention layers (Q, K, V projections + output projection)
        for _ in range(4):  # 4 linear layers per attention
            cpu_result = cpu_sim.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length)
            cpu_total_time += cpu_result['time_ms']
            cpu_total_energy += cpu_result['energy_mj']
        
        # FFN layers (2 linear layers)
        cpu_result1 = cpu_sim.simulate_linear_layer(hidden_size, ffn_size, batch_size * seq_length)
        cpu_result2 = cpu_sim.simulate_linear_layer(ffn_size, hidden_size, batch_size * seq_length)
        cpu_total_time += cpu_result1['time_ms'] + cpu_result2['time_ms']
        cpu_total_energy += cpu_result1['energy_mj'] + cpu_result2['energy_mj']
    
    # Apply minimum inference time
    cpu_total_time = max(cpu_total_time, min_inference_time_ms)
    
    results['cpu'] = {
        'latency_ms': cpu_total_time,
        'energy_mj': cpu_total_energy,
        'throughput_tokens_per_sec': batch_size * seq_length / (cpu_total_time / 1000),
        'power_watts': cpu_sim.specs.power_consumption,
        'platform': 'CPU'
    }
    
    # GPU Performance (Original Model Only - no compression)
    print("Simulating GPU performance...")
    gpu_total_time = 0
    gpu_total_energy = 0
    
    # Add GPU initialization overhead (realistic for inference)
    gpu_init_time = 2.0  # 2ms GPU initialization/context setup
    gpu_init_energy = gpu_init_time / 1000 * gpu_sim.specs.idle_power
    gpu_total_time += gpu_init_time
    gpu_total_energy += gpu_init_energy
    
    # Simulate each layer type on GPU
    for layer in range(num_layers):
        # Attention layers (Q, K, V projections + output projection)
        for _ in range(4):  # 4 linear layers per attention
            gpu_result = gpu_sim.simulate_linear_layer(hidden_size, hidden_size, batch_size * seq_length)
            gpu_total_time += gpu_result['time_ms']
            gpu_total_energy += gpu_result['energy_mj']
        
        # FFN layers (2 linear layers)
        gpu_result1 = gpu_sim.simulate_linear_layer(hidden_size, ffn_size, batch_size * seq_length)
        gpu_result2 = gpu_sim.simulate_linear_layer(ffn_size, hidden_size, batch_size * seq_length)
        gpu_total_time += gpu_result1['time_ms'] + gpu_result2['time_ms']
        gpu_total_energy += gpu_result1['energy_mj'] + gpu_result2['energy_mj']
    
    # Apply minimum inference time
    gpu_total_time = max(gpu_total_time, min_inference_time_ms)
    
    results['gpu'] = {
        'latency_ms': gpu_total_time,
        'energy_mj': gpu_total_energy,
        'throughput_tokens_per_sec': batch_size * seq_length / (gpu_total_time / 1000),
        'power_watts': gpu_sim.specs.power_consumption,
        'platform': 'GPU'
    }
    
    # FPGA Performance (Both Original and Compressed)
    print("Simulating FPGA performance...")
    
    # Original model on FPGA
    fpga_original = fpga_sim.simulate_transformer(
        num_layers, hidden_size, ffn_size, num_heads, seq_length, batch_size, is_compressed=False
    )
    
    # Compressed model on FPGA
    fpga_compressed = fpga_sim.simulate_transformer(
        num_layers, hidden_size, ffn_size, num_heads, seq_length, batch_size, is_compressed=True
    )
    
    results['fpga_original'] = {
        'latency_ms': fpga_original['latency_ms'],
        'energy_mj': fpga_original['energy_mj'],
        'throughput_tokens_per_sec': fpga_original['throughput_tokens_per_sec'],
        'power_watts': fpga_sim.specs.power_consumption,
        'platform': 'FPGA (Original)',
        'dsp_utilization': fpga_original['dsp_utilization'],
        'bram_utilization_mb': fpga_original['bram_utilization_mb']
    }
    
    results['fpga_compressed'] = {
        'latency_ms': fpga_compressed['latency_ms'],
        'energy_mj': fpga_compressed['energy_mj'],
        'throughput_tokens_per_sec': fpga_compressed['throughput_tokens_per_sec'],
        'power_watts': fpga_sim.specs.power_consumption,
        'platform': 'FPGA (Compressed)',
        'dsp_utilization': fpga_compressed['dsp_utilization'],
        'bram_utilization_mb': fpga_compressed['bram_utilization_mb'],
        'compression_rate': f"{block_size}x"
    }
    
    # Calculate relative performance metrics
    baseline_latency = results['cpu']['latency_ms']
    baseline_energy = results['cpu']['energy_mj']
    
    for platform in results:
        results[platform]['latency_speedup'] = baseline_latency / results[platform]['latency_ms']
        results[platform]['energy_efficiency'] = baseline_energy / results[platform]['energy_mj']
        results[platform]['energy_per_token'] = results[platform]['energy_mj'] / (batch_size * seq_length)
    
    # Summary comparison
    results['summary'] = {
        'fastest_platform': min(results.keys(), key=lambda x: results[x]['latency_ms'] if x != 'summary' else float('inf')),
        'most_energy_efficient': min(results.keys(), key=lambda x: results[x]['energy_mj'] if x != 'summary' else float('inf')),
        'highest_throughput': max(results.keys(), key=lambda x: results[x]['throughput_tokens_per_sec'] if x != 'summary' else 0),
        'compression_benefit': {
            'latency_improvement': results['fpga_original']['latency_ms'] / results['fpga_compressed']['latency_ms'],
            'energy_improvement': results['fpga_original']['energy_mj'] / results['fpga_compressed']['energy_mj'],
            'throughput_improvement': results['fpga_compressed']['throughput_tokens_per_sec'] / results['fpga_original']['throughput_tokens_per_sec']
        }
    }
    
    return results


def print_comparison_results(results):
    """
    Print comprehensive comparison results in a formatted way.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PLATFORM COMPARISON RESULTS")
    print("="*80)
    
    # Platform comparison table
    platforms = ['cpu', 'gpu', 'fpga_original', 'fpga_compressed']
    
    print(f"\n{'Platform':<20} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'Throughput':<15} {'Power (W)':<12}")
    print("-" * 80)
    
    for platform in platforms:
        if platform in results:
            r = results[platform]
            print(f"{r['platform']:<20} {r['latency_ms']:<15.2f} {r['energy_mj']:<15.2f} "
                  f"{r['throughput_tokens_per_sec']:<15.0f} {r['power_watts']:<12.1f}")
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    summary = results['summary']
    print(f"Fastest Platform: {results[summary['fastest_platform']]['platform']}")
    print(f"Most Energy Efficient: {results[summary['most_energy_efficient']]['platform']}")
    print(f"Highest Throughput: {results[summary['highest_throughput']]['platform']}")
    
    print(f"\nCompression Benefits on FPGA:")
    cb = summary['compression_benefit']
    print(f"  Latency Improvement: {cb['latency_improvement']:.2f}x")
    print(f"  Energy Improvement: {cb['energy_improvement']:.2f}x")
    print(f"  Throughput Improvement: {cb['throughput_improvement']:.2f}x")
    
    print("\n" + "="*80)
    print("SPEEDUP vs CPU BASELINE")
    print("="*80)
    
    for platform in platforms:
        if platform in results:
            r = results[platform]
            print(f"{r['platform']:<20} {r['latency_speedup']:<15.2f}x {r['energy_efficiency']:<15.2f}x")
    
    if 'fpga_compressed' in results:
        print(f"\nFPGA Resource Utilization (Compressed Model):")
        r = results['fpga_compressed']
        print(f"  DSP Utilization: {r['dsp_utilization']*100:.1f}%")
        print(f"  BRAM Usage: {r['bram_utilization_mb']:.1f} MB") 