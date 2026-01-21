import os
import csv

# mapping from emotion string to numeric id
emotion_mapping = {'ang': 0, 'neu': 1, 'hap': 2, 'sad': 3, 'fru': 4, 'exc': 5}

dataset = 'iemocap'
# data split
data = 'valid'

# input file paths
utterance_file = './datasets/iemocap/csv/'+dataset+'_'+data+'_utterances.tsv'
emotion_file = './datasets/iemocap/csv/'+dataset+'_'+data+'_emotion.tsv'
speaker_file = './datasets/iemocap/csv/'+dataset+'_'+data+'_speakers.tsv'

# output directory
output_dir = './datasets/iemocap'
os.makedirs(output_dir, exist_ok=True)

utterances_output = os.path.join(output_dir, dataset+'_'+data+'_'+'utterances.tsv')
speakers_output = os.path.join(output_dir, dataset+'_'+data+'_'+'speakers.tsv')
emotion_output = os.path.join(output_dir, dataset+'_'+data+'_'+'emotion.tsv')
classify_output = os.path.join(output_dir, dataset+'_'+data+'_'+'classify.tsv')

# initialize data containers
utterances_data = {}
emotions_data = {}
speakers_data = {}

# read and process utterance file
with open(utterance_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        dialogue_id = parts[0]
        utterances = parts[1].split(' __eou__ ')
        utterances_data[dialogue_id] = utterances

# read and process emotion file
with open(emotion_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        dialogue_id = parts[0]
        emotions = parts[1].split(',')
        emotions = [emotion_mapping.get(e.strip(), 5) for e in emotions]  # map emotions to numeric ids
        emotions_data[dialogue_id] = emotions

# read and process speaker file
with open(speaker_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        dialogue_id = parts[0]
        speakers = parts[1].split(',')
        speaker_mapping = {}
        speaker_index = 0
        new_speakers = []
        for s in speakers:
            speaker_id = s.strip().split('_')[-1]
            # use first two characters as prefix
            prefix = speaker_id[:2]
            if prefix not in speaker_mapping:
                speaker_mapping[prefix] = speaker_index
                speaker_index += 1
            new_speakers.append(speaker_mapping[prefix])
        speakers_data[dialogue_id] = new_speakers
  
# sanitize special characters in strings
def process_special_chars(data):
    return [str(item).replace('\t', ' ').replace('\n', '') for item in data]
# write utterances.tsv
with open(utterances_output, 'w') as f:
    for dialogue_id, utterances in utterances_data.items():
        utterances = process_special_chars(utterances)
        line = '\t'.join([dialogue_id] + utterances) + '\n'
        f.write(line)

# write speakers.tsv
with open(speakers_output, 'w') as f:
    for dialogue_id, speakers in speakers_data.items():
        speakers = process_special_chars(speakers)
        line = '\t'.join([dialogue_id] + speakers) + '\n'
        f.write(line)

# write emotion.tsv
with open(emotion_output, 'w') as f:
    for dialogue_id, emotions in emotions_data.items():
        emotions = process_special_chars(emotions)
        line = '\t'.join([dialogue_id] + emotions) + '\n'
        f.write(line)

# write classify.tsv (same content as emotion.tsv)
with open(classify_output, 'w') as f:
    for dialogue_id, emotions in emotions_data.items():
        emotions = process_special_chars(emotions)
        line = '\t'.join([dialogue_id] + emotions) + '\n'
        f.write(line)


# write classify.tsv using csv writer (same as emotion.tsv)
with open(classify_output, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    for dialogue_id, emotions in emotions_data.items():
        row = [dialogue_id] + emotions
        writer.writerow(row)

print(f'Files generated successfully:\n- {utterances_output}\n- {speakers_output}\n- {emotion_output}\n- {classify_output}')