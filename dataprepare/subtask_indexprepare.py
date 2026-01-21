import numpy as np
import csv
import os

if __name__ == '__main__':
    # initialize lists to hold utterances and speaker info
    utterances = []
    speakers = []
    # dataset name
    dataset = 'iemocap'
    # data split (here 'valid' refers to the validation split)
    data = 'valid'

    # build base file path using f-strings for readability
    file_base = f'./datasets/{dataset}/{dataset}_{data}_'
    # construct specific file paths for utterances, speakers and subtask index
    file_utterances = f'{file_base}utterances.tsv'
    file_speakers = f'{file_base}speakers.tsv'
    file_subtask = f'{file_base}subtask01_index.tsv'

    # read utterances file; use list comprehension for concise code
    with open(file_utterances, 'r', encoding='utf-8') as f:
        utterances = [line.strip().split('\t')[1:] for line in f]

    # read speakers file; use list comprehension for concise code
    with open(file_speakers, 'r', encoding='utf-8') as f:
        speakers = [line.strip().split('\t')[1:] for line in f]

    # remove existing subtask01 index file if present
    if os.path.exists(file_subtask):
        os.remove(file_subtask)

    # iterate over each dialogue
    for j, conv in enumerate(speakers):
        # get unique speakers in the current dialogue
        unique_speakers = np.unique(conv)
        # initialize a mapping from speaker to their first occurrence index
        speaker_memo = {speaker: conv.index(speaker) for speaker in unique_speakers}

        # initialize result row and add dialogue identifier
        res = [f'te_{j}']

        # iterate speakers in the dialogue
        for i, curr_speaker in enumerate(conv):
            # get last seen index for current speaker
            last_index = speaker_memo[curr_speaker]
            # update last seen index for current speaker
            speaker_memo[curr_speaker] = i

            if i == last_index:
                # first occurrence -> append -1
                res.append(-1)
            else:
                # append previous occurrence index
                res.append(last_index)

        # append the processed row to subtask index file
        with open(file_subtask, 'a', newline='', encoding='utf-8') as f_output:
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(res)