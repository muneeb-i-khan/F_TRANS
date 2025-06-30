import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class BlockCirculantMatrixCompression:
    """
    Implements Block Circulant Matrix (BCM) compression for Transformer models.
    BCM compression reduces the parameter size by representing weight matrices
    as block circulant matrices, which can be efficiently implemented on FPGAs.
    """
    
    def __init__(self, block_size=4):
        """
        Initialize the BCM compression with a specific block size.
        
        Args:
            block_size: Size of each block in the block circulant matrix.
        """
        self.block_size = block_size
    
    def _create_circulant_matrix(self, first_row):
        """
        Create a circulant matrix from the first row.
        
        Args:
            first_row: The first row of the circulant matrix
            
        Returns:
            A circulant matrix
        """
        n = len(first_row)
        circulant = torch.zeros(n, n, dtype=first_row.dtype, device=first_row.device)
        
        for i in range(n):
            circulant[i] = torch.roll(first_row, i)
            
        return circulant
    
    def _find_best_circulant_approximation(self, block):
        """
        Find the best circulant matrix approximation for a given block.
        This uses least squares to find the optimal first row.
        
        Args:
            block: The matrix block to approximate
            
        Returns:
            The optimal first row for circulant approximation
        """
        n = block.shape[0]
        if n != block.shape[1]:
            # Handle non-square blocks by padding
            max_dim = max(block.shape)
            padded_block = torch.zeros(max_dim, max_dim, device=block.device, dtype=block.dtype)
            padded_block[:block.shape[0], :block.shape[1]] = block
            block = padded_block
            n = max_dim
        
        # Create circulant matrix construction matrix
        # Each row represents how to construct one row of the circulant matrix from the first row
        A = torch.zeros(n * n, n, device=block.device, dtype=block.dtype)
        for i in range(n):
            for j in range(n):
                # For circulant matrix, element (i,j) comes from position (j-i) % n of the first row
                A[i * n + j, (j - i) % n] = 1.0
        
        # Flatten the target block
        b = block.flatten()
        
        # Solve least squares: A @ x = b, where x is the optimal first row
        try:
            # Use torch.linalg.lstsq for better numerical stability
            solution = torch.linalg.lstsq(A, b, driver='gels')
            optimal_first_row = solution.solution[:n]
        except:
            # Fallback to pseudo-inverse if lstsq fails
            A_pinv = torch.pinverse(A)
            optimal_first_row = A_pinv @ b
            optimal_first_row = optimal_first_row[:n]
        
        return optimal_first_row

    def _compress_matrix(self, matrix):
        """
        Compress a matrix using block circulant representation.
        
        Args:
            matrix: The weight matrix to compress
            
        Returns:
            compressed_matrix: A compressed representation (optimal first row of each block)
            original_shape: The original shape of the matrix
        """
        original_shape = matrix.shape
        rows, cols = original_shape
        
        # Adjust dimensions to be divisible by block_size
        padded_rows = ((rows + self.block_size - 1) // self.block_size) * self.block_size
        padded_cols = ((cols + self.block_size - 1) // self.block_size) * self.block_size
        
        # Pad the matrix if needed
        if padded_rows > rows or padded_cols > cols:
            padded_matrix = torch.zeros(padded_rows, padded_cols, device=matrix.device, dtype=matrix.dtype)
            padded_matrix[:rows, :cols] = matrix
            matrix = padded_matrix
        
        # Extract optimal first row of each block using least squares approximation
        num_blocks_rows = padded_rows // self.block_size
        num_blocks_cols = padded_cols // self.block_size
        compressed_data = []
        
        for i in range(num_blocks_rows):
            for j in range(num_blocks_cols):
                block = matrix[i*self.block_size:(i+1)*self.block_size, 
                               j*self.block_size:(j+1)*self.block_size]
                
                # Find the best circulant approximation for this block
                optimal_first_row = self._find_best_circulant_approximation(block)
                compressed_data.append(optimal_first_row)
        
        compressed_data = torch.stack(compressed_data)
        
        return {
            'compressed_data': compressed_data,
            'original_shape': original_shape,
            'block_size': self.block_size,
            'num_blocks_rows': num_blocks_rows,
            'num_blocks_cols': num_blocks_cols
        }
    
    def _decompress_matrix(self, compressed_data, original_shape=None):
        """
        Decompress a matrix from its block circulant representation.
        
        Args:
            compressed_data: Dictionary containing compressed matrix data
            original_shape: The original shape to restore
            
        Returns:
            The decompressed matrix
        """
        data = compressed_data['compressed_data']
        block_size = compressed_data['block_size']
        num_blocks_rows = compressed_data['num_blocks_rows']
        num_blocks_cols = compressed_data['num_blocks_cols']
        
        # Reconstruct the padded matrix
        padded_rows = num_blocks_rows * block_size
        padded_cols = num_blocks_cols * block_size
        padded_matrix = torch.zeros(padded_rows, padded_cols, device=data.device)
        
        block_idx = 0
        for i in range(num_blocks_rows):
            for j in range(num_blocks_cols):
                first_row = data[block_idx]
                block = self._create_circulant_matrix(first_row)
                padded_matrix[i*block_size:(i+1)*block_size, 
                              j*block_size:(j+1)*block_size] = block
                block_idx += 1
        
        # If original shape provided, trim the matrix
        if original_shape:
            rows, cols = original_shape
            decompressed_matrix = padded_matrix[:rows, :cols]
        else:
            decompressed_matrix = padded_matrix
        
        return decompressed_matrix
    
    def compress_layer(self, layer):
        """
        Compress the weights of a linear layer using block circulant matrices.
        If the layer appears to be the final classification head (i.e., has 2 output
        features for binary classification), leave it uncompressed to avoid
        accuracy degradation because its size is negligible.
        """
        # Skip non-linear layers or very small classifier heads
        if not isinstance(layer, nn.Linear):
            return layer

        # Do not compress classification head (typically tiny, e.g., 2 outputs)
        if layer.out_features == 2:
            return layer
            
        compressed_layer = deepcopy(layer)
        weight_data = layer.weight.data
        
        # Store original weights for backward pass
        compressed_layer.original_weight = weight_data.clone()
        
        # Compress the weight matrix
        compressed_layer.compressed_weight = self._compress_matrix(weight_data)
        
        # For inference, replace the weight with the decompressed version
        decompressed_weight = self._decompress_matrix(compressed_layer.compressed_weight)
        compressed_layer.weight.data = decompressed_weight
        
        return compressed_layer
    
    def compress_model(self, model):
        """
        Compress all linear layers in a model.
        
        Args:
            model: A PyTorch model
            
        Returns:
            A compressed version of the model
        """
        compressed_model = deepcopy(model)
        
        # Recursively compress all linear layers in the model
        for name, module in compressed_model.named_children():
            if len(list(module.children())) > 0:
                # If module has children, recursively compress them
                compressed_submodule = self.compress_model(module)
                setattr(compressed_model, name, compressed_submodule)
            elif isinstance(module, nn.Linear):
                # If module is a linear layer, compress it
                compressed_module = self.compress_layer(module)
                setattr(compressed_model, name, compressed_module)
        
        return compressed_model
    
    def compute_compression_rate(self, model, compressed_model):
        """
        Compute the compression rate achieved by BCM compression.
        
        Args:
            model: Original PyTorch model
            compressed_model: Compressed PyTorch model
            
        Returns:
            Compression rate (original size / compressed size)
        """
        original_params = sum(p.numel() for p in model.parameters())
        
        # Count parameters in compressed model (only the first rows of blocks)
        compressed_params = 0
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'compressed_weight'):
                compressed_params += module.compressed_weight['compressed_data'].numel()
                # Add bias parameters if they exist
                if module.bias is not None:
                    compressed_params += module.bias.numel()
            elif isinstance(module, nn.Linear) and not hasattr(module, 'compressed_weight'):
                # For any uncompressed linear layers
                compressed_params += sum(p.numel() for p in module.parameters())
            elif not isinstance(module, nn.Linear) and not any(isinstance(submodule, nn.Linear) for submodule in module.modules()):
                # For non-linear layers that don't contain linear sublayers
                compressed_params += sum(p.numel() for p in module.parameters())
        
        return original_params / compressed_params if compressed_params > 0 else 0


def compress_transformer(model, block_size=4):
    """
    Compress a transformer model using Block Circulant Matrix compression.
    
    Args:
        model: The transformer model to compress
        block_size: Size of each block in the block circulant matrix
        
    Returns:
        compressed_model: The compressed model
        compression_rate: The achieved compression rate
    """
    compressor = BlockCirculantMatrixCompression(block_size=block_size)
    compressed_model = compressor.compress_model(model)
    compression_rate = compressor.compute_compression_rate(model, compressed_model)
    
    return compressed_model, compression_rate 