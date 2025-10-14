# src/ddg_predictor/feature_extraction/boltz_wrapper.py

import subprocess
import logging
import sys
from types import SimpleNamespace

# A library module should not configure the root logger.
# The executable script (03_extract_features.py) handles this.

def run_boltz_prediction(queries_dir: str, config: SimpleNamespace) -> None:
    """
    Builds and runs the Boltz prediction command, capturing output for error reporting.
    """
    logging.info(f"Starting Boltz prediction for queries in: {queries_dir}")
    
    # Use sys.executable to ensure the script runs with the same Python interpreter
    cmd = [sys.executable, "-m", "boltz_mod.main", "predict", queries_dir]

    # Build the command with flags from the config
    boltz_flags = getattr(config, "boltz_flags", {})
    for key, value in boltz_flags.items():
        flag = key if key.startswith("--") else f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])

    command_str = " ".join(cmd)
    logging.info(f"Executing command: {command_str}")
    
    # Using subprocess.run is simpler if real-time streaming is not needed.
    # We capture output to show it only in case of an error, reducing console clutter.
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError automatically on non-zero exit codes
        )
        # Log the full stdout only if needed for debugging, or keep it clean
        # logging.debug(f"Boltz STDOUT:\n{process.stdout}")

    except FileNotFoundError:
        logging.error(f"Command not found. Is '{sys.executable}' correct and is Boltz installed in the environment?")
        raise
    except subprocess.CalledProcessError as e:
        logging.error("Boltz execution failed.")
        logging.error(f"Return Code: {e.returncode}")
        # Provide the detailed output from Boltz since it failed
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        raise e
    
    logging.info("Boltz execution finished successfully.")