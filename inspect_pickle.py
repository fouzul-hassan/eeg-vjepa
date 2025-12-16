"""Quick inspection script for ZuCo pickle files"""
import pickle

pickle_path = r"C:\MSc Files\MSc Project\E2T-w-VJEPA\e2t-cloned-amirhojati\eeg-vjepa\src\datasets\ZuCo\task2-NR\pickle\task2-NR-dataset-spectro.pickle"

print("Loading pickle...")
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print("Subjects:", list(data.keys()))
first_subj = list(data.keys())[0]
print("First subject:", first_subj)
print("Num sentences:", len(data[first_subj]))

# Get first valid sentence
for i, sent in enumerate(data[first_subj]):
    if sent is not None:
        raw = sent['sentence_level_EEG']['rawData']
        print(f"Sample {i} - Shape: {raw.shape}, dtype: {raw.dtype}")
        print(f"Channels: {raw.shape[0]}, Time: {raw.shape[1]}")
        break

# Count all valid samples
total = 0
for subj in data.values():
    for sent in subj:
        if sent is not None:
            total += 1
print(f"Total valid samples: {total}")
