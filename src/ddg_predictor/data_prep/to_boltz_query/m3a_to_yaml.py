#!/usr/bin/env python3
import os
import re
import argparse
import yaml

class A3MtoYAMLConverter:
    """
    Converts .a3m MSA files into YAML files based on a template.
    Expects template structure like:
    sequences:
      - protein:
          id: ...
          sequence: ...
          msa: ...
    """

    def __init__(self, input_path, output_dir, template_file):
        self.input_path = input_path
        self.output_dir = os.path.join(output_dir, 'boltz_queries/')

        self.template_file = template_file
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_msa_query_info(self, a3m_file):
        """Extracts query sequence ID and ungapped sequence from first record in A3M."""
        with open(a3m_file, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        lines = [ln.rstrip('\n') for ln in raw_lines]

        header = None
        seq_lines = []
        for i, line in enumerate(lines):
            if line.startswith('>'):
                header = line[1:].strip()
                j = i + 1
                while j < len(lines) and not lines[j].startswith('>'):
                    seq_lines.append(lines[j].strip())
                    j += 1
                break

        if header is None:
            raise ValueError(f"No query sequence found in {a3m_file}")

        raw_seq = ''.join(seq_lines)
        seq = re.sub(r'[\.\-]', '', raw_seq)
        seq = re.sub(r'[^A-Za-z]', '', seq).upper()

        seq_id = os.path.splitext(os.path.basename(a3m_file))[0]
        return seq_id, seq

    def _update_template_with_values(self, data, seq_id, sequence, msa_path):
        """
        Update the YAML template with sequence information.
        Assumes template has a 'sequences' list containing dicts with 'protein'.
        """
        if not isinstance(data, dict):
            data = {}

        if 'sequences' not in data or not isinstance(data['sequences'], list):
            data['sequences'] = []

        updated = False
        for entry in data['sequences']:
            if 'protein' in entry and isinstance(entry['protein'], dict):
                entry['protein']['id'] = '[A]'
                entry['protein']['sequence'] = sequence
                entry['protein']['msa'] = msa_path
                updated = True
                break

        if not updated:
            # If no entry existed, create a new one
            data['sequences'].append({
                'protein': {
                    'id': seq_id,
                    'sequence': sequence,
                    'msa': msa_path
                }
            })

        return data

    def convert_one(self, a3m_file):
        """Convert a single .a3m file to YAML using the template."""
        try:
            seq_id, sequence = self.extract_msa_query_info(a3m_file)
        except Exception as e:
            print(f"Skipping {a3m_file}: {e}")
            return

        msa_path = a3m_file

        with open(self.template_file, 'r', encoding='utf-8') as tf:
            try:
                data = yaml.safe_load(tf) or {}
            except Exception as e:
                print(f"Error loading template {self.template_file}: {e}")
                return

        new_data = self._update_template_with_values(data, seq_id, sequence, msa_path)

        safe_name = re.sub(r'[^\w\-\_\.]', '_', seq_id)
        out_path = os.path.join(self.output_dir, f"{safe_name}.yaml")

        with open(out_path, 'w', encoding='utf-8') as out_f:
            yaml.safe_dump(new_data, out_f, sort_keys=False, default_flow_style=False)



    def batch_convert(self):
        """Convert all .a3m files in a directory (or a single file)."""
        if os.path.isfile(self.input_path):
            if self.input_path.endswith('.a3m'):
                self.convert_one(self.input_path)
            return

        for root, _, files in os.walk(self.input_path):

            for f in files:
                if f.endswith('.a3m'):
                    a3m_path = os.path.join(root, f)
                    self.convert_one(a3m_path)


def main():
    parser = argparse.ArgumentParser(description="Convert .a3m files into YAML files using a template")
    parser.add_argument("input_dir", help="Directory with .a3m files or a single .a3m file")
    parser.add_argument("output_dir", help="Directory where the YAML files will be saved")
    parser.add_argument("template", help="YAML template file")
    args = parser.parse_args()


    print(args.input_dir)
    print(args.output_dir)

    conv = A3MtoYAMLConverter(args.input_dir, args.output_dir, args.template)
    conv.batch_convert()


if __name__ == "__main__":
    main()
