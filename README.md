# ASR_Benchmark
Benchmark ARS models on your dataset

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asr_benchmark.git
   cd asr_benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Create a configuration file (`configs/config.yaml`).
2. Run the evaluation:
   ```bash
   python scripts/evaluate.py -c configs/config.yaml
   ```

The predictions will be saved in the `predictions` directroy.
