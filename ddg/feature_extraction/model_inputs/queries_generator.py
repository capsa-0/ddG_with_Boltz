"""
Module: MsaToBoltzYamlConverter
Description: Converts MSA files to Boltz-compatible YAML query format.
"""

import os
import re
import glob
import logging
import yaml
from pathlib import Path
from tqdm import tqdm

from ddg.datasets.ids import sanitize_id

logger = logging.getLogger(__name__)


class IndentDumper(yaml.SafeDumper):
    """Custom YAML dumper for consistent indentation."""
    def increase_indent(self, flow=False, indentless=False):
        """Override indentation for better formatting."""
        return super().increase_indent(flow, False)

    
class MsaToBoltzYamlConverter:
    """Convert MSA files to Boltz YAML queries."""

    def __init__(self, config):
        """
        Initialize converter with project configuration.
        
        Args:
            config: ProjectConfig object from config_loader
        """
        self.config = config
        logger.debug("Initialized MsaToBoltzYamlConverter")

    # ----- A3M to YAML Conversion Methods -----

    def _extract_msa_query_info(self, a3m_file: str) -> tuple[str, str]:
        """
        Extract query sequence ID and ungapped sequence from A3M file.
        
        Args:
            a3m_file: Path to A3M format file
            
        Returns:
            Tuple of (sequence_id, ungapped_sequence)
        """
        with open(a3m_file, 'r') as f:
            lines = [line.strip() for line in f]
        
        header = lines[0][1:]
        
        # ----- Collect sequence lines until next header -----
        seq_lines = []
        for i in range(1, len(lines)):
            if lines[i].startswith('>'):
                break
            seq_lines.append(lines[i])
        
        # ----- Remove gap characters (dots and dashes) -----
        ungapped_seq = re.sub(r'[\.\-]', '', ''.join(seq_lines))
        return header, ungapped_seq

    def _build_query_doc(self, seq_id: str, sequence: str, msa_path: str) -> dict:
        """
        Build Boltz YAML query dictionary.
        
        Args:
            seq_id: Sequence identifier
            sequence: Protein sequence
            msa_path: Path to MSA file (absolute or relative)
            
        Returns:
            Dictionary in Boltz YAML format
        """
        # ----- Single-sequence mode: tell Boltz to run without an MSA -----
        if getattr(self.config, "no_msa", False):
            msa_to_use = "empty"
        else:
            # ----- Convert to relative path from project root -----
            msa_path_obj = Path(msa_path).resolve()
            try:
                msa_relative = msa_path_obj.relative_to(Path.cwd())
                msa_to_use = str(msa_relative)
            except ValueError:
                logger.debug(f"MSA path outside cwd, using absolute: {msa_path}")
                msa_to_use = str(msa_path)
        
        return {
            "sequences": [{
                "protein": {
                    "id": seq_id,
                    "sequence": sequence,
                    "msa": msa_to_use
                }
            }]
        }

    def convert_a3m_dir_to_yaml(self):
        """
        Convert all A3M files in directory to Boltz YAML queries.
        Saves to config.queries_dir with sanitized filenames.
        """
        msa_dir = self.config.msa_dir
        yaml_output_dir = self.config.queries_dir
        
        os.makedirs(yaml_output_dir, exist_ok=True)
        a3m_files = sorted(glob.glob(os.path.join(msa_dir, "*.a3m")))
        
        if not a3m_files:
            logger.warning(f"No .a3m files found in {msa_dir}")
            return

        logger.info(f"Converting {len(a3m_files)} MSA files to YAML format")

        for idx, a3m_path in enumerate(tqdm(a3m_files, desc="Converting A3M to YAML"), start=1):
            try:
                original_id, sequence = self._extract_msa_query_info(a3m_path)
                doc = self._build_query_doc(str(idx), sequence, a3m_path)
                
                # ----- Sanitize original ID for filename (shared sanitizer) -----
                safe_name = sanitize_id(original_id)
                out_path = os.path.join(yaml_output_dir, f"{safe_name}.yaml")

                with open(out_path, 'w') as out_f:
                    out_f.write(f"# Original ID: {original_id}\n")
                    yaml.dump(
                        doc,
                        out_f,
                        Dumper=IndentDumper,
                        sort_keys=False,
                        default_flow_style=False,
                        indent=2,
                    )
                logger.debug(f"Converted {original_id} to {out_path}")
            except Exception as e:
                logger.error(f"Failed to convert {os.path.basename(a3m_path)}: {e}")

