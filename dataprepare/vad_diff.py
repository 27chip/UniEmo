import numpy as np  # numpy for numerical operations
import csv  # csv module for TSV read/write
import os  # os module for filesystem operations

if __name__ == '__main__':  # execute when run as a script
    v_values, a_values, d_values, speakers = [], [], [], []  # lists to store v/a/d values and speaker info
    dataset = 'iemocap'  # dataset name
    data = 'test'  # data split

    # build base file path for the dataset split
    file_base = './datasets/' + dataset + '/' + dataset + '_' + data + '_'

    # define full paths to valence/arousal/dominance and speakers files
    file_v = file_base + 'valence.tsv'
    file_a = file_base + 'arousal.tsv'
    file_d = file_base + 'dominance.tsv'
    file_speakers = file_base + 'speakers.tsv'

    # define output diff file paths
    file_result_v = file_base + 'v_diff.tsv'
    file_result_a = file_base + 'a_diff.tsv'
    file_result_d = file_base + 'd_diff.tsv'

    # remove existing result files to avoid duplicate appends
    for file in [file_result_v, file_result_a, file_result_d]:
        if os.path.exists(file):
            os.remove(file)

    # read valence file, skip first column, store remaining columns per dialogue
    with open(file_v) as f:
        for line in f:
            content = line.strip().split('\t')[1:]  # strip whitespace and split line
            v_values.append(content)

    # read arousal file, skip first column, store remaining columns per dialogue
    with open(file_a) as f:
        for line in f:
            content = line.strip().split('\t')[1:]
            a_values.append(content)

    # read dominance file, skip first column, store remaining columns per dialogue
    with open(file_d) as f:
        for line in f:
            content = line.strip().split('\t')[1:]
            d_values.append(content)

    # read speakers file, skip first column, store per dialogue
    with open(file_speakers) as f:
        for line in f:
            content = line.strip().split('\t')[1:]
            speakers.append(content)

    # iterate over speakers for each dialogue
    for j, conv in enumerate(speakers):
        unique_speakers = np.unique(conv)
        speaker_memo = {}

        for unique_speaker in unique_speakers:
            speaker_memo[unique_speaker] = conv.index(unique_speaker)

        res_v = []
        res_a = []
        res_d = []
        res_v.append('te_' + str(j))
        res_a.append('te_' + str(j))
        res_d.append('te_' + str(j))

        for i, curr_speaker in enumerate(conv):
            last_index = speaker_memo[curr_speaker]
            speaker_memo[curr_speaker] = i

            if i == last_index:
                # first occurrence -> zero diff
                res_v.append(0)
                res_a.append(0)
                res_d.append(0)
            else:
                last_v = float(v_values[j][last_index])
                curr_v = float(v_values[j][i])
                res_v.append(round(curr_v - last_v, 4))

                last_a = float(a_values[j][last_index])
                curr_a = float(a_values[j][i])
                res_a.append(round(curr_a - last_a, 4))

                last_d = float(d_values[j][last_index])
                curr_d = float(d_values[j][i])
                res_d.append(round(curr_d - last_d, 4))

        with open(file_result_v, 'a', newline='') as f_result_v, \
                open(file_result_a, 'a', newline='') as f_result_a, \
                open(file_result_d, 'a', newline='') as f_result_d:
            tsv_result_v = csv.writer(f_result_v, delimiter='\t')
            tsv_result_a = csv.writer(f_result_a, delimiter='\t')
            tsv_result_d = csv.writer(f_result_d, delimiter='\t')
            tsv_result_v.writerow(res_v)
            tsv_result_a.writerow(res_a)
            tsv_result_d.writerow(res_d)

    print(f"Generated v_diff file: {file_result_v}")
    print(f"Generated a_diff file: {file_result_a}")
    print(f"Generated d_diff file: {file_result_d}")