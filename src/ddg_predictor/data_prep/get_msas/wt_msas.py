from mmseq2_boltz import run_mmseqs2
from Bio import SeqIO
import os


def get_sequences_from_fasta(fasta_path: str) -> tuple[list[str], list[str]]:
    """
    Args:
        fasta_path (str): Path to the FASTA file.
    Returns:
        tuple[list[str], list[str]]: 
            - List of sequence IDs. 
            - List of corresponding sequences as strings.
    """
    # Parse a FASTA file and extract IDs and sequences
    ids = []
    sequences = []
    
    with open(fasta_path, 'r') as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))
    
    return ids, sequences


def generate_msas_from_fasta(fasta_path: str, output_dir: str, **kwargs) -> None:
    """
    Args:
        fasta_path (str): Path to the input FASTA file (can be multifasta).
        output_dir (str): Directory where the MSA results will be stored.
        **kwargs: Extra keyword arguments passed to run_mmseqs2.
    Returns:
        None
    """
    # Read sequences from the FASTA file
    ids, seqs = get_sequences_from_fasta(fasta_path)

    # Generate an MSA for each sequence
    for seq_id, sequence in zip(ids, seqs):

        a3m_lines = run_mmseqs2(x=sequence, prefix=f"tmp_{seq_id}", **kwargs)  # Call external MMseqs2 wrapper
        
        # Take the first MSA result and replace the header with the sequence ID
        msa_content = a3m_lines[0]
        msa_content = replace_first_header(msa_content, seq_id)

        # Build output path and write to .a3m file
        output_path = os.path.join(output_dir, f"{seq_id}.a3m")
        with open(output_path, 'w') as f:
            f.write(msa_content)



def replace_first_header(msa_content: str, new_id: str) -> str:
    """
    Args:
        msa_content (str): MSA file content in A3M format.
        new_id (str): New sequence identifier to replace the first FASTA header.
    Returns:
        str: Modified MSA content with the updated first header.
    """
    # Replace the first header line in the MSA with a new ID
    lines = msa_content.split('\n')
    if lines and lines[0].startswith('>'):
        lines[0] = f'>{new_id}'
    return '\n'.join(lines)



