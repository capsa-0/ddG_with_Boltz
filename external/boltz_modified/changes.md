Summary of Changes

1. Added --embeddings_only flag (main.py)
New CLI flag that skips diffusion and structure prediction
Automatically enables --write_embeddings when set
Also skips affinity predictions when combined with structure prediction
2. Updated model prediction
Boltz2 (boltz2.py:1080): Modified predict_step to conditionally add coordinates only if they exist
Boltz1 (boltz1.py:1169): Same conditional logic for coordinates
Both models now pass skip_run_structure=embeddings_only so the model skips the entire diffusion sampling process
3. Enhanced writer (writer.py:49)
Now handles missing coordinates gracefully
When in embeddings-only mode, it skips structure file writing and only outputs the embeddings NPZ files
Early returns prevent unnecessary processing
Usage
To extract only embeddings without structure prediction:

boltz predict --data your_input.fasta --out_dir ./results --embeddings_only

This will:

Skip all diffusion sampling
Skip structure prediction
Only compute the s and z embeddings
Save embeddings to embeddings_{id}.npz files
Be significantly faster than full predictions
The embeddings will be saved in the predictions directory with the format: embeddings_{record_id}.npz containing both s and z arrays.
