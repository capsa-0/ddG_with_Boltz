"""
Module: run_boltz
Description: Execute Boltz structure prediction on query YAML files.
Usage:
    python run_boltz.py <experiment_config.yaml> [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ddg.config.config_loader import ProjectConfig

# ----- Setup logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_boltz_predictions(config) -> None:
    """
    Execute Boltz predictions for all query YAML files.
    
    Args:
        config: ProjectConfig instance containing paths and Boltz parameters
        
    Raises:
        FileNotFoundError: If queries directory not found
        RuntimeError: If Boltz prediction fails
    """
    # ----- Get queries directory -----
    queries_dir = config.queries_dir
    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")
    
    logger.info(f"Found queries directory: {queries_dir}")
    
    # ----- Get Boltz parameters from config -----
    boltz_flags = config.boltz_flags
    logger.info(f"Boltz configuration: {boltz_flags}")
    
    # ----- Prepare output directory -----
    output_path = Path(config.exp_processed_dir) 
    logger.info(f"Output directory: {output_path}")
    
    # ----- Build Boltz CLI command -----
    logger.info(f"Preparing Boltz predictions for all YAML files in {queries_dir}")
    
    cmd = [
        "boltz",
        "predict",
        str(queries_dir),
        "--out_dir", str(output_path),
        "--cache", boltz_flags.get("cache", "~/.boltz"),
        "--accelerator", boltz_flags.get("accelerator", "gpu"),
        "--recycling_steps", str(boltz_flags.get("recycling_steps", 3)),
        "--model", boltz_flags.get("model", "boltz2"),
        "--write_embeddings",
        "--embeddings_only"
    ]
    
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Boltz prediction failed with return code {result.returncode}")
    
    # ----- Rename output directory to standard location -----
    default_boltz_output_dir = output_path / f"boltz_results_{config._dirs['queries_dir']}"
    new_boltz_output_dir = Path(config.raw_features_dir)
    if default_boltz_output_dir.exists():
        default_boltz_output_dir.rename(new_boltz_output_dir)
        logger.info(f"Renamed Boltz output: {default_boltz_output_dir} -> {new_boltz_output_dir}")

    logger.info(f"Boltz predictions completed! Results saved to {output_path}")