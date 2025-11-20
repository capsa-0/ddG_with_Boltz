import subprocess
import logging
import sys
from types import SimpleNamespace

def run_boltz_prediction(queries_dir: str, config: SimpleNamespace) -> None:
    """
    Builds and runs the Boltz prediction command.
    MODIFIED FOR DEBUGGING: capture_output is False to see real-time logs.
    """
    logging.info(f"Starting Boltz prediction for queries in: {queries_dir}")
    
    # Use sys.executable to ensure the script runs with the same Python interpreter
    cmd = [sys.executable, "-m", "boltz.main", "predict", queries_dir]

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
    
    try:
        # CAMBIO IMPORTANTE: capture_output=False
        # Esto permite que veas el output de Boltz directamente en la consola
        # mientras se ejecuta. Si falta un flag o hay un warning, lo verás ahí.
        subprocess.run(
            cmd,
            capture_output=False, 
            text=True,
            check=True
        )

    except FileNotFoundError:
        logging.error(f"Command not found. Is '{sys.executable}' correct and is Boltz installed in the environment?")
        raise

    except subprocess.CalledProcessError as e:
        logging.error("Boltz execution failed.")
        logging.error(f"Return Code: {e.returncode}")
        # Como capture_output=False, el error ya se imprimió en la consola arriba.
        logging.error("Check the console output above for detailed error messages from Boltz.")
        raise e
    
    logging.info("Boltz execution finished successfully.")