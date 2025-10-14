#!/bin/bash

# Este comando asegura que el script se detenga inmediatamente si algún comando falla.
# Es una práctica esencial para los pipelines.
set -e

# --- Ejecución del Pipeline de Datos y Características ---


python3 scripts/01_prepare_dataset.py

python3 scripts/02_generate_boltz_queries.py

python3 scripts/03_extract_features.py

python3 scripts/04_clean_outputs.py

python3 scripts/05_process_features.py

python3 scripts/06_analyze_features.py

echo "Pipeline completed successfully."