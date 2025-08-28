import argparse
import pandas as pd
import os

from wt_msas import generate_msas_from_fasta
from mut_msa import A3mMutator  

def main() -> None:
    """
    Main script to generate MSAs from a multifasta file and then apply mutations
    to all generated MSAs.
    """
    parser = argparse.ArgumentParser(description='Generate MSAs from multifasta and apply mutations')
    parser.add_argument('--input_fasta', required=True, help='Input multifasta file path')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated A3M files')
    parser.add_argument('--mutations_csv', required=True, help='CSV file with mutations (sequence_id, mutation, ddg)')
    
    args = parser.parse_args()

    msa_output_dir = os.path.join(args.output_dir, "msas")
    os.makedirs(msa_output_dir, exist_ok=True)

    # Step 1: Generate MSAs from the input multifasta
    generate_msas_from_fasta(args.input_fasta, msa_output_dir)
    print(f"MSA generation completed. Files saved in: {args.output_dir}")

    # Step 2: Load mutations and apply to all MSAs in the directory
    print("Applying mutations to generated MSAs...")
    mutations_df = pd.read_csv(args.mutations_csv)
    mutator = A3mMutator(msa_output_dir, mutations_df)
    mutator.mutate_directory()
    print("Mutation application completed.")


if __name__ == '__main__':
    main()
