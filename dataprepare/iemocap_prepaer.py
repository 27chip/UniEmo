import os
import csv
import re
import chardet

# dataset and split definition
dataset = 'iemocap'
data = 'train'

# input file path templates
dialogue_path = "./datasets/iemocap/csv/Session{}/dialog/transcriptions"
emotion_path = "./datasets/iemocap/csv/Session{}/dialog/EmoEvaluation"

# output directory (create if not exists)
output_dir = './datasets/iemocap'
os.makedirs(output_dir, exist_ok=True)

# define the six output file paths
output_dialogue_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'utterances.tsv')
output_speaker_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'speakers.tsv')
output_emotion_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'emotion.tsv')
output_valence_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'valence.tsv')
output_arousal_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'arousal.tsv')
output_dominance_csv = os.path.join(output_dir, dataset + '_' + data + '_' + 'dominance.tsv')

# emotion mapping dictionary
emotion_mapping = {
    'ang': 0, 'neu': 1, 'dis': 2, 'fea': 3,
    'hap': 4, 'sad': 5, 'sur': 6, 'fru': 7,
    'exc': 8, 'xxx': 9
}

# helper: delete file if it exists
def check_and_delete_file(file_path):
    if os.path.exists(file_path):
        print(f"File exists, deleting: {file_path}")
        os.remove(file_path)

# initialize CSV file with header columns
def init_csv(file_path, columns):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)

# process data for a single session
def process_session(session):
    dialogue_dir = dialogue_path.format(session)
    emotion_dir = emotion_path.format(session)

    # containers for rows collected from files
    dialogue_rows = []
    speaker_rows = []
    emotion_rows = []
    valence_rows = []
    arousal_rows = []
    dominance_rows = []

    # iterate over all transcript files in the dialogue directory
    dialogue_files = [f for f in os.listdir(dialogue_dir) if f.endswith('.txt')]
    for dialogue_file in dialogue_files:
        dialogue_id = dialogue_file.split('.')[0]

        # detect file encoding for robust reading
        with open(os.path.join(dialogue_dir, dialogue_file), 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        # read dialogue transcript
        with open(os.path.join(dialogue_dir, dialogue_file), 'r', encoding=file_encoding) as f:
            dialogue_lines = f.readlines()

        # read corresponding emotion annotation file
        emotion_file = os.path.join(emotion_dir, dialogue_file)
        with open(emotion_file, 'r', encoding=file_encoding) as f:
            emotion_lines = f.readlines()

        # initialize lists to store data for current file
        utterances = []
        speakers = []
        emotions = []
        valences = []
        arousals = []
        dominances = []

        # parse dialogue lines
        for line in dialogue_lines:
            # try to match different transcript line formats
            # impro type
            match = re.match(r"^(Ses\d{2}[FM]_impro\d{2}_[FM]\d{3}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)
            if not match:
                # Session3_impro05a/b
                match = re.match(r"^(Ses\d{2}[FM]_impro\d{2}[ab]_[FM]\d{3}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)
                if not match:
                    # Session5 M/FXX0
                    match = re.match(r"^(Ses\d{2}[FM]_impro\d{2}_[FM]XX\d{1}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)
                    if not match:
                            # script type
                        match = re.match(r"^(Ses\d{2}[FM]_script\d{2}_\d{1}_[FM]\d{3}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)
                        if not match:
                            # Session3_impro05a/b
                            match = re.match(r"^(Ses\d{2}[FM]_script\d{2}_\d{1}[ab]_[FM]\d{3}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)
                            if not match:
                                # Session5 M/FXX0
                                match = re.match(r"^(Ses\d{2}[FM]_script\d{2}_\d{1}_[FM]XX\d{1}) \[(\d+\.\d+)-(\d+\.\d+)\]: (.*)$", line)

            if match:
                speaker = match.group(1)
                utterance = match.group(4)
                if utterance:
                    # set default emotion and VAD dims
                    emotion = "xxx"
                    valence = arousal = dominance = 2.5000

                    # find matching emotion annotation for this turn
                    for emotion_line in emotion_lines:
                        emotion_match = re.match(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\t(\S+)\t(\S+)\t\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]", emotion_line)
                        if emotion_match:
                            turn_name = emotion_match.group(3)
                            if turn_name == speaker:
                                emotion = emotion_match.group(4)
                                valence = float(emotion_match.group(5))
                                arousal = float(emotion_match.group(6))
                                dominance = float(emotion_match.group(7))
                                break

                    # filter out unwanted emotion labels
                    if emotion not in ['dis', 'fea', 'sur', 'xxx']:
                        utterances.append(utterance)
                        speakers.append(speaker)
                        emotions.append(emotion)
                        valences.append(valence)
                        arousals.append(arousal)
                        dominances.append(dominance)

        # assemble rows for the current dialogue file
        if utterances and speakers:
            dialogue_row = [dialogue_id, " __eou__ ".join(utterances)]
            speaker_row = [dialogue_id, ", ".join(speakers)]
            emotion_row = [dialogue_id, ",".join(emotions)]
            valence_row = [dialogue_id, ",".join(map(str, valences))]
            arousal_row = [dialogue_id, ",".join(map(str, arousals))]
            dominance_row = [dialogue_id, ",".join(map(str, dominances))]

            # append assembled rows to global lists
            dialogue_rows.append(dialogue_row)
            speaker_rows.append(speaker_row)
            emotion_rows.append(emotion_row)
            valence_rows.append(valence_row)
            arousal_rows.append(arousal_row)
            dominance_rows.append(dominance_row)

    return dialogue_rows, speaker_rows, emotion_rows, valence_rows, arousal_rows, dominance_rows

# global containers for all sessions
all_dialogue_rows = []
all_speaker_rows = []
all_emotion_rows = []
all_valence_rows = []
all_arousal_rows = []
all_dominance_rows = []


# choose which sessions to process
# to process Session1 to Session4
for session in range(1, 5):
    dialogue_rows, speaker_rows, emotion_rows, valence_rows, arousal_rows, dominance_rows = process_session(session)
    all_dialogue_rows.extend(dialogue_rows)
    all_speaker_rows.extend(speaker_rows)
    all_emotion_rows.extend(emotion_rows)
    all_valence_rows.extend(valence_rows)
    all_arousal_rows.extend(arousal_rows)
    all_dominance_rows.extend(dominance_rows)

'''
# to process only Session5
session = 5
dialogue_rows, speaker_rows, emotion_rows, valence_rows, arousal_rows, dominance_rows = process_session(session)
all_dialogue_rows.extend(dialogue_rows)
all_speaker_rows.extend(speaker_rows)
all_emotion_rows.extend(emotion_rows)
all_valence_rows.extend(valence_rows)
all_arousal_rows.extend(arousal_rows)
all_dominance_rows.extend(dominance_rows)
'''

# sort rows of each file by dialogue ID
all_dialogue_rows = sorted(all_dialogue_rows, key=lambda x: x[0])
all_speaker_rows = sorted(all_speaker_rows, key=lambda x: x[0])
all_emotion_rows = sorted(all_emotion_rows, key=lambda x: x[0])
all_valence_rows = sorted(all_valence_rows, key=lambda x: x[0])
all_arousal_rows = sorted(all_arousal_rows, key=lambda x: x[0])
all_dominance_rows = sorted(all_dominance_rows, key=lambda x: x[0])

# write sorted rows to output files
def write_sorted_data(file_path, rows):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in rows:
            writer.writerow(row)

write_sorted_data(output_dialogue_csv, all_dialogue_rows)
write_sorted_data(output_speaker_csv, all_speaker_rows)
write_sorted_data(output_emotion_csv, all_emotion_rows)
write_sorted_data(output_valence_csv, all_valence_rows)
write_sorted_data(output_arousal_csv, all_arousal_rows)
write_sorted_data(output_dominance_csv, all_dominance_rows)

print("Data processing complete. Files sorted by dialogue ID and saved to:")
print(output_dialogue_csv)
print(output_speaker_csv)
print(output_emotion_csv)
print(output_valence_csv)
print(output_arousal_csv)
print(output_dominance_csv)