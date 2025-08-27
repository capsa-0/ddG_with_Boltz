#!/usr/bin/env python3
import os
import re
import argparse
import yaml

class A3MtoYAMLConverter:
    def __init__(self, input_path, output_dir, template_file):

        self.input_path = input_path
        self.output_dir = output_dir
        self.template_file = template_file
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_msa_query_info(self, a3m_file):

        with open(a3m_file, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        lines = [ln.rstrip('\n') for ln in raw_lines]

        header = None
        seq_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('>'):
                header = line[1:].strip()
                j = i + 1
                while j < len(lines) and not lines[j].lstrip().startswith('>'):
                    seq_lines.append(lines[j].strip())
                    j += 1
                break
            i += 1

        if header is None:
            return

        raw_seq = ''.join(seq_lines)
        seq = re.sub(r'[\.\-]', '', raw_seq)
        seq = re.sub(r'[^A-Za-z]', '', seq).upper()

        seq_id = os.path.relpath(a3m_file).split('/')[-1].rsplit('.', 1)[0]

        return seq_id, seq

    def _update_template_with_values(self, data, id_code, sequence, msa_path):

        if isinstance(data, dict) and 'sequences' in data and isinstance(data['sequences'], list):
            updated = False
            for entry in data['sequences']:
                if isinstance(entry, dict):
                    for k, v in entry.items():
                        if isinstance(v, dict):

                            v['id'] = id_code
                            v['sequence'] = sequence
                            if 'msa' in v:
                                v['msa'] = msa_path
                            elif 'msa_path' in v:
                                v['msa_path'] = msa_path
                            else:
                                v['msa'] = msa_path
                            updated = True
                            break
                    if updated:
                        break
            if updated:
                return data

        if isinstance(data, dict):
            data['id'] = id_code
            data['sequence'] = sequence
            data['msa_path'] = msa_path
            return data

        return {
            'id': id_code,
            'sequence': sequence,
            'msa_path': msa_path
        }

    def convert_one(self, a3m_file):
        try:
            id_code, sequence = self.extract_msa_query_info(a3m_file)
        except Exception as e:
            return

        msa_path = a3m_file

        with open(self.template_file, 'r', encoding='utf-8') as tf:
            try:
                data = yaml.safe_load(tf) or {}
            except Exception as e:
                return

        new_data = self._update_template_with_values(data, id_code, sequence, msa_path)

        safe_name = re.sub(r'[^\w\-\_\.]', '_', id_code)
        out_path = os.path.join(self.output_dir, f"{safe_name}.yaml")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            yaml.safe_dump(new_data, out_f, sort_keys=False, default_flow_style=False)

        print(f"Created: {out_path}")

    def batch_convert(self):
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
    parser = argparse.ArgumentParser(description="Converts .a3m files into .yaml files using a template")
    parser.add_argument("input", help="Directory with .a3m files or a single .a3m file")
    parser.add_argument("output", help="Directory where the YAML files will be saved")
    parser.add_argument("template", help="YAML template file")

    args = parser.parse_args()

    conv = A3MtoYAMLConverter(args.input, args.output, args.template)
    conv.batch_convert()


if __name__ == "__main__":
    main()
