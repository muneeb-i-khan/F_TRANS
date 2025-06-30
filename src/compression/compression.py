import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class EnhancedBCMLayer(nn.Module):
    """
    Enhanced Block Circulant Matrix Layer with FFT-based computation.
    Implements the enhanced BCM formulation from the paper.
    """
    
    def __init__(self, in_features, out_features, block_size=4, bias=True):
        super(EnhancedBCMLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # Calculate number of blocks
        self.f = (out_features + block_size - 1) // block_size  # f = m ÷ b
        self.g = (in_features + block_size - 1) // block_size   # g = n ÷ b
        
        # Padded dimensions
        self.padded_out_features = self.f * block_size
        self.padded_in_features = self.g * block_size
        
        # Index vectors for each block (enhanced formulation)
        # Shape: (f, g, block_size) - one index vector per block
        self.index_vectors = nn.Parameter(torch.randn(self.f, self.g, block_size))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        # Initialize index vectors
        nn.init.xavier_uniform_(self.index_vectors.view(-1, self.block_size))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _fft_circular_conv(self, index_vector, x_block):
        """
        Perform FFT-based circular convolution: p ⊛ x = IFFT(FFT(p) ⊙ FFT(x))
        
        Args:
            index_vector: Index vector of shape (block_size,)
            x_block: Input block of shape (batch_size, block_size)
            
        Returns:
            Result of circular convolution of shape (batch_size, block_size)
        """
        batch_size = x_block.shape[0]
        
        # FFT of index vector and input block
        # Use real FFT for efficiency since our data is real
        fft_p = torch.fft.fft(index_vector)  # (block_size,)
        fft_x = torch.fft.fft(x_block, dim=1)  # (batch_size, block_size)
        
        # Element-wise multiplication in frequency domain
        fft_result = fft_p.unsqueeze(0) * fft_x  # (batch_size, block_size)
        
        # IFFT to get back to time domain
        result = torch.fft.ifft(fft_result, dim=1).real  # (batch_size, block_size)
        
        return result
    
    def forward(self, x):
        """
        Forward pass using enhanced BCM with FFT-based computation.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Pad input if necessary
        if x.shape[1] < self.padded_in_features:
            x_padded = F.pad(x, (0, self.padded_in_features - x.shape[1]))
        else:
            x_padded = x
        
        # Reshape input into blocks: x = [x₁ᵀ, x₂ᵀ, ..., xₘᵀ]ᵀ
        x_blocks = x_padded.view(batch_size, self.g, self.block_size)  # (batch_size, g, block_size)
        
        # Initialize output
        output_blocks = torch.zeros(batch_size, self.f, self.block_size, 
                                  device=x.device, dtype=x.dtype)
        
        # Perform BCM computation for each block
        for i in range(self.f):
            for j in range(self.g):
                # Get index vector for this block
                p_ij = self.index_vectors[i, j]  # (block_size,)
                
                # Get input block
                x_j = x_blocks[:, j, :]  # (batch_size, block_size)
                
                # Perform FFT-based circular convolution: W_ij * x_j = p_ij ⊛ x_j
                conv_result = self._fft_circular_conv(p_ij, x_j)  # (batch_size, block_size)
                
                # Accumulate result
                output_blocks[:, i, :] += conv_result
        
        # Reshape output back to (batch_size, out_features)
        output = output_blocks.view(batch_size, self.padded_out_features)
        
        # Trim to original output size
        if self.padded_out_features > self.out_features:
            output = output[:, :self.out_features]
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output


class BlockCirculantMatrixCompression:
    """
    Enhanced Block Circulant Matrix (BCM) compression for Transformer models.
    Implements the enhanced BCM formulation with proper index vector computation.
    """
    
    def __init__(self, block_size=4):
        """
        Initialize the BCM compression with a specific block size.
        
        Args:
            block_size: Size of each block in the block circulant matrix (FFT size).
        """
        self.block_size = block_size
    
    def _compute_enhanced_index_vector(self, block):
        """
        Compute the enhanced index vector according to Equation (3) in the paper:
        p_ij = [1/b * Σ(W_1j), 1/b * Σ(W_2j), ..., 1/b * Σ(W_bj)]
        
        Args:
            block: The weight matrix block of shape (block_size, block_size)
            
        Returns:
            Enhanced index vector of shape (block_size,)
        """
        # Compute average of each row: p_ij[k] = 1/b * Σ(W_kj) for j=1 to b
        index_vector = torch.mean(block, dim=1)  # Average across columns for each row
        return index_vector
    
    def _create_circulant_matrix_from_index(self, index_vector):
        """
        Create a circulant matrix from the enhanced index vector.
        
        Args:
            index_vector: The enhanced index vector
            
        Returns:
            A circulant matrix
        """
        n = len(index_vector)
        circulant = torch.zeros(n, n, dtype=index_vector.dtype, device=index_vector.device)
        
        for i in range(n):
            circulant[i] = torch.roll(index_vector, i)
            
        return circulant
    
    def _compress_matrix(self, matrix):
        """
        Compress a matrix using enhanced block circulant representation.
        
        Args:
            matrix: The weight matrix to compress (out_features, in_features)
            
        Returns:
            Dictionary containing compressed representation
        """
        original_shape = matrix.shape
        rows, cols = original_shape
        
        # Adjust dimensions to be divisible by block_size
        padded_rows = ((rows + self.block_size - 1) // self.block_size) * self.block_size
        padded_cols = ((cols + self.block_size - 1) // self.block_size) * self.block_size
        
        # Pad the matrix if needed
        if padded_rows > rows or padded_cols > cols:
            padded_matrix = torch.zeros(padded_rows, padded_cols, 
                                      device=matrix.device, dtype=matrix.dtype)
            padded_matrix[:rows, :cols] = matrix
            matrix = padded_matrix
        
        # Calculate number of blocks
        f = padded_rows // self.block_size  # f = m ÷ b
        g = padded_cols // self.block_size  # g = n ÷ b
        
        # Extract enhanced index vectors for each block
        index_vectors = torch.zeros(f, g, self.block_size, 
                                  device=matrix.device, dtype=matrix.dtype)
        
        for i in range(f):
            for j in range(g):
                block = matrix[i*self.block_size:(i+1)*self.block_size, 
                             j*self.block_size:(j+1)*self.block_size]
                
                # Compute enhanced index vector for this block
                index_vectors[i, j] = self._compute_enhanced_index_vector(block)
        
        return {
            'index_vectors': index_vectors,
            'original_shape': original_shape,
            'block_size': self.block_size,
            'f': f,  # number of row blocks
            'g': g   # number of column blocks
        }
    
    def _decompress_matrix(self, compressed_data):
        """
        Decompress a matrix from its enhanced block circulant representation.
        
        Args:
            compressed_data: Dictionary containing compressed matrix data
            
        Returns:
            The decompressed matrix
        """
        index_vectors = compressed_data['index_vectors']
        original_shape = compressed_data['original_shape']
        block_size = compressed_data['block_size']
        f = compressed_data['f']
        g = compressed_data['g']
        
        # Reconstruct the padded matrix
        padded_rows = f * block_size
        padded_cols = g * block_size
        padded_matrix = torch.zeros(padded_rows, padded_cols, 
                                  device=index_vectors.device, dtype=index_vectors.dtype)
        
        for i in range(f):
            for j in range(g):
                # Get index vector for this block
                index_vector = index_vectors[i, j]
                
                # Create circulant matrix from index vector
                circulant_block = self._create_circulant_matrix_from_index(index_vector)
                
                # Place in padded matrix
                padded_matrix[i*block_size:(i+1)*block_size, 
                            j*block_size:(j+1)*block_size] = circulant_block
        
        # Trim to original shape
        rows, cols = original_shape
        decompressed_matrix = padded_matrix[:rows, :cols]
        
        return decompressed_matrix
    
    def compress_layer(self, layer):
        """
        Compress a linear layer using enhanced BCM.
        
        Args:
            layer: A PyTorch Linear layer
            
        Returns:
            An enhanced BCM layer
        """
        # Skip non-linear layers
        if not isinstance(layer, nn.Linear):
            return layer
        
        # Skip very small layers (like classification heads)
        if layer.out_features <= 2:
            return layer
        
        # Create enhanced BCM layer
        bcm_layer = EnhancedBCMLayer(
            in_features=layer.in_features,
            out_features=layer.out_features,
            block_size=self.block_size,
            bias=(layer.bias is not None)
        )
        
        # Compress the weight matrix and extract index vectors
        compressed_data = self._compress_matrix(layer.weight.data)
        
        # Set the index vectors in the BCM layer
        bcm_layer.index_vectors.data = compressed_data['index_vectors']
        
        # Copy bias if it exists
        if layer.bias is not None:
            bcm_layer.bias.data = layer.bias.data.clone()
        
        # Store compression info for analysis
        bcm_layer.compression_info = compressed_data
        
        return bcm_layer
    
    def compress_model(self, model):
        """
        Compress all linear layers in a model using enhanced BCM.
        
        Args:
            model: A PyTorch model
            
        Returns:
            A compressed version of the model
        """
        compressed_model = deepcopy(model)
        
        # Recursively compress all linear layers
        def compress_module(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Compress this linear layer
                    compressed_layer = self.compress_layer(child)
                    setattr(module, name, compressed_layer)
                else:
                    # Recursively process child modules
                    compress_module(child)
        
        compress_module(compressed_model)
        return compressed_model
    
    def compute_compression_rate(self, model, compressed_model):
        """
        Compute the compression rate achieved by enhanced BCM compression.
        
        Args:
            model: Original PyTorch model
            compressed_model: Compressed PyTorch model
            
        Returns:
            Compression rate (original size / compressed size)
        """
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        return original_params / compressed_params if compressed_params > 0 else 0


def compress_transformer(model, block_size=4):
    """
    Compress a transformer model using Enhanced Block Circulant Matrix compression.
    
    Args:
        model: The transformer model to compress
        block_size: Size of each block in the block circulant matrix (FFT size)
        
    Returns:
        compressed_model: The compressed model with enhanced BCM layers
        compression_rate: The achieved compression rate
    """
    compressor = BlockCirculantMatrixCompression(block_size=block_size)
    compressed_model = compressor.compress_model(model)
    compression_rate = compressor.compute_compression_rate(model, compressed_model)
    
    return compressed_model, compression_rate 