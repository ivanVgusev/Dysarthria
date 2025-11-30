import pandas as pd
import os

df = pd.read_csv("dataset/data.csv")

def extract_speaker_id(path):
    filename = os.path.basename(path)
    speaker_id = filename.split('_')[0]
    return speaker_id


def change_paths(path):
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))

    fpath_changed = os.path.join("dataset", dirname, filename)
    return fpath_changed
df["filename"] = df["filename"].apply(change_paths)
df.insert(2, 'speaker_id', df['filename'].apply(extract_speaker_id))

df.to_csv("dataset/data_more_info.csv")
