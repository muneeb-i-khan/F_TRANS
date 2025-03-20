# FTRANS: Energy-Efficient Acceleration of Transformers using FPGA

This project implements FTRANS, an energy-efficient approach for accelerating Transformer models using FPGA. The implementation focuses on:

1. **Model Compression**: Compressing Transformer models (like BERT) using block-based compression methods.
2. **Performance Evaluation**: Training and evaluating the compressed models on the IMDB sentiment classification dataset.
3. **FPGA Simulation**: Simulating the execution of compressed models on FPGA hardware.

## Project Structure

- `src/model/`: Transformer model definitions
- `src/compression/`: Implementation of model compression techniques
- `src/training/`: Scripts for training and evaluating models
- `src/fpga_sim/`: FPGA simulator for accelerated transformer execution

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Model Compression**:
   ```
   python src/compression/compress.py
   ```

2. **Training**:
   ```
   python src/training/train.py
   ```

3. **FPGA Simulation**:
   ```
   python src/fpga_sim/simulator.py
   ```

## Results

Performance metrics and energy efficiency comparisons between original and compressed models running on FPGA will be documented in the results folder after simulation. 

# To run the entire pipeline
python main.py --run-all

# To run individual steps
python main.py --compress  # Only compress the model
python main.py --train     # Only train the models
python main.py --simulate  # Only run the simulation

# To configure model parameters
python main.py --hidden-size 768 --num-layers 12 --num-heads 12 --block-size 8 --run-all 