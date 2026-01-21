import csv
import os


"""
Generate loss-mask files where each utterance participates in loss computation.

The main function `generate_all_ones_mask` reads an utterances TSV and
writes a corresponding mask TSV where each utterance is marked with '1'.
Each output row is prefixed with an identifier `te_{idx}` followed by mask
values to align with downstream readers.
"""

def generate_all_ones_mask(input_file, output_file):
    """
    Create an all-ones loss mask file so every utterance contributes to loss.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        tsv_writer = csv.writer(f_out, delimiter='\t')
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            # remove first column (id) and take the remaining utterances
            utterances = line.split('\t')[1:]
            # generate a '1' mask for each utterance
            mask = ['1'] * len(utterances)
            # prefix with an identifier to mirror other dataset files
            result = [f'te_{idx}'] + mask
            tsv_writer.writerow(result)


# example invocation
dataset = 'iemocap'
#data_splits = ['train', 'dev', 'test']
data_splits = ['train','valid']


for split in data_splits:
    utterances_file = f'./datasets/{dataset}/{dataset}_{split}_utterances.tsv'
    mask_file = f'./datasets/{dataset}/{dataset}_{split}_loss_mask.tsv'
    generate_all_ones_mask(utterances_file, mask_file)

print("Done")