# ddG Prediction with Boltz Embeddings

A machine learning pipeline for predicting protein stability changes (ΔΔG) upon mutation using structural embeddings from the Boltz-2 protein structure prediction model.

## Overview

This project leverages deep learning embeddings from [Boltz-2](https://github.com/jwohlwend/boltz) to predict the effect of single-point mutations on protein stability. The pipeline extracts rich structural representations from predicted protein structures and uses them as features for ΔΔG prediction.

## Features

- **Automated Data Processing**: Handles standard mutation datasets (UniProt ID, mutation, ΔΔG)
- **MSA Generation & Mutation**: Creates multiple sequence alignments and applies mutations
- **Boltz Integration**: Extracts embeddings (single, pair, and distance representations)
- **Feature Engineering**: Computes difference tensors and aggregates them into ML-ready features
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Flexible Configuration**: YAML-based configuration for easy experimentation

## Installation

### Requirements

- Python 3.10-3.13
- CUDA-compatible GPU (optional, but recommended for faster processing)
- [Boltz-2 model weights](https://github.com/jwohlwend/boltz) (see caching section)

### Setup

1. Clone the repository:
```bash
git clone <repository_url>
cd ddG_with_Boltz
```

2. Create and activate a virtual environment:
```bash
python -m venv venv_ddg_boltz
source venv_ddg_boltz/bin/activate  # On Linux/Mac
# or
venv_ddg_boltz\Scripts\activate  # On Windows
```

3. Install the package:
```bash
pip install -e .
```

For CUDA support, install additional dependencies:
```bash
pip install -e ".[cuda]"
```

## Project Structure

```
ddG_with_Boltz/
├── config/
│   ├── params.yaml              # Main configuration file
│   └── boltz_query_template.yaml
├── data/
│   ├── raw/                     # Input datasets (CSV format)
│   └── processed/               # Generated MSAs and queries
├── scripts/
│   ├── 01_prepare_dataset.py    # Data loading and preprocessing
│   ├── 02_generate_boltz_queries.py  # MSA generation and mutation
│   ├── 03_extract_features.py   # Boltz embedding extraction
│   ├── 05_process_features.py   # Feature aggregation and analysis
│   └── 07_train_model.py        # Model training (WIP)
├── src/
│   └── ddg_predictor/
│       ├── data_processing/     # Data loaders and MSA handlers
│       ├── feature_extraction/  # Boltz wrapper
│       └── feature_processing/  # Embedding difference and analysis
├── boltz/                       # Boltz-2 model implementation
├── boltz_cache/                 # Model checkpoints cache
└── *_results/                   # Output embeddings and predictions
```

## Usage

The pipeline consists of sequential steps, each configurable via `config/params.yaml`.

### Step 1: Prepare Dataset

Loads raw mutation data and prepares it for processing.

```bash
python scripts/01_prepare_dataset.py
```

**Input**: CSV file with columns: `uniprot`, `mut`, `ddg`  
**Output**: `data/processed/{dataset_name}/` with processed sequences and mutations

### Step 2: Generate Boltz Queries

Creates MSAs for wild-type proteins and applies mutations.

```bash
python scripts/02_generate_boltz_queries.py
```

**Output**: YAML query files for Boltz in `data/processed/{dataset_name}/boltz_queries/`

### Step 3: Extract Features with Boltz

Runs Boltz-2 to generate embeddings for wild-type and mutant proteins.

```bash
python scripts/03_extract_features.py
```

**Configuration options**:
- `process_one_by_one`: Process queries individually (safer for large datasets)
- `accelerator`: "cpu" or "gpu"
- `recycling_steps`: Number of recycling iterations (0-3+)
- `write_embeddings`: Must be `true` to extract features

**Output**: Embedding tensors (`.npz` files) in `{out_dir}/{protein_id}/`
- `s.npz`: Single representation (per-residue features)
- `z.npz`: Pair representation (residue-residue interactions)
- `pdistogram.npz`: Predicted distance distributions

### Step 5: Process Features

Computes difference tensors between WT and mutant, aggregates them into scalar features, and generates correlation plots.

```bash
python scripts/05_process_features.py
```

**Output**:
- `{out_dir}/{protein_id}_{mutation}/diff_*.npz`: Difference tensors
- `{out_dir}/features_summary.csv`: Aggregated features for ML
- `{out_dir}/correlation_plots/`: Scatter plots of features vs. ΔΔG

## Configuration

Edit `config/params.yaml` to customize the pipeline:

```yaml
data_processing:
  raw_data_path: "data/raw/your_dataset.csv"
  max_msa_sequences: 1000  # Number of MSA sequences to use

feature_extraction:
  process_one_by_one: false  # Process all queries in batch
  boltz_flags:
    accelerator: "gpu"      # "cpu" or "gpu"
    model: "boltz2"
    recycling_steps: 3      # 0-5+ (higher = more accurate but slower)
    cache: "boltz_cache"    # Path to model checkpoints
    out_dir: "results"      # Output directory
```

## Feature Engineering

The pipeline extracts two types of features:

### 1. **Difference Tensors** (`EmbeddingDiffer`)
Computes the difference between wild-type and mutant embeddings:
- **Absolute difference**: `|E_mut - E_wt|`
- **Signed difference**: `E_mut - E_wt`
- **L2 norm**: `||E_mut - E_wt||_2`

### 2. **Aggregated Features** (`FeatureAnalyzer`)
Applies statistical aggregations to difference tensors:

**Global features**: Computed over the entire protein
- Mean, Max, Std, Sum
- Entropy (Shannon entropy)
- Gini coefficient

**Local features**: Computed only at the mutation site
- Same aggregations applied to residue-specific slices
- For pair representations: diagonal element `diff[pos, pos, :]`

## Data Format

### Input CSV
```csv
uniprot,mut,ddg
P12345,A10V,0.5
P12345,G25D,-1.2
```

- `uniprot`: UniProt accession code
- `mut`: Mutation in format `{original_aa}{position}{new_aa}` (e.g., "A10V")
- `ddg`: Experimental ΔΔG value (kcal/mol)

### Output Features CSV
```csv
sequence_id,mutation,ddg,global_s_mean,global_s_max,...,local_z_entropy,...
P12345,A10V,0.5,0.023,0.187,...,1.45,...
```

Each row contains:
- Metadata (protein ID, mutation, experimental ΔΔG)
- Global and local aggregations for each embedding type (s, z, pdistogram)

## Development

### Adding New Aggregators

Edit `src/ddg_predictor/feature_processing/feature_analyzer.py`:

```python
self.AGGREGATORS = {
    "mean": lambda arr: np.mean(arr),
    "custom_metric": lambda arr: your_custom_function(arr),
    # ... add more
}
```

### Modifying Boltz Parameters

The Boltz wrapper accepts all standard Boltz CLI arguments through `boltz_flags` in the config.

## Citation

If you use this pipeline, please cite:

- **Boltz-2**: [Paper/GitHub link]
- **Your work**: [If applicable]

## License

[Specify license]

## Troubleshooting

### Out of Memory (OOM) Errors
- Set `process_one_by_one: true` in config
- Reduce `max_msa_sequences`
- Use CPU instead of GPU (slower but more memory)

### Missing Embeddings
- Ensure `write_embeddings: true` in Boltz flags
- Check that Boltz completed successfully (look for `.npz` files in output)

### Shape Mismatches
- Verify that WT and mutant sequences have the same length
- Check that mutations are within valid residue range

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

[Your contact information]
