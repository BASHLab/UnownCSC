import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

def process_wesad():
    project_root = os.getcwd() 
    raw_dir = os.path.join(project_root, 'data', 'WESAD')
    out_dir = os.path.join(project_root, 'data', 'WESAD_Processed')
    
    print(f"📁 Looking for raw data in: {raw_dir}")
    print(f"📁 Saving processed chunks to: {out_dir}")
    
    if not os.path.exists(raw_dir):
        print("❌ ERROR: Could not find the raw WESAD folder.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # The config names and their sampling rates
    freqs = {
        'chest_ACC': 700, 'chest_ECG': 700, 'chest_EMG': 700, 'chest_RESP': 700, 'chest_EDA': 700, 'chest_TEMP': 700,
        'wrist_ACC': 32, 'wrist_BVP': 64, 'wrist_EDA': 4, 'wrist_TEMP': 4
    }

    # 🚨 THE FIX: Map MAESTRO's config names to the exact WESAD .pkl case-sensitive keys
    WESAD_PKL_KEYS = {
        'chest_ACC': ('chest', 'ACC'),
        'chest_ECG': ('chest', 'ECG'),
        'chest_EMG': ('chest', 'EMG'),
        'chest_EDA': ('chest', 'EDA'),
        'chest_TEMP': ('chest', 'Temp'),  # .pkl uses 'Temp'
        'chest_RESP': ('chest', 'Resp'),  # .pkl uses 'Resp'
        'wrist_ACC': ('wrist', 'ACC'),
        'wrist_BVP': ('wrist', 'BVP'),
        'wrist_EDA': ('wrist', 'EDA'),
        'wrist_TEMP': ('wrist', 'TEMP')   # Wrist actually uses 'TEMP'
    }

    window_sec = 8
    shift_sec = 2 

    for subj in range(2, 18):
        subj_str = f'S{subj}'
        pkl_path = os.path.join(raw_dir, subj_str, f'{subj_str}.pkl')
        
        if not os.path.exists(pkl_path):
            continue
            
        print(f"\n⚙️ Loading massive .pkl file for {subj_str} (This takes a few seconds)...")
        subj_out = os.path.join(out_dir, subj_str)
        os.makedirs(subj_out, exist_ok=True)

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        signals = data['signal']
        labels = data['label']

        chest_pts = window_sec * 700
        chest_shift = shift_sec * 700
        
        saved_count = 0
        skipped_count = 0
        
        pbar = tqdm(range(0, len(labels) - chest_pts + 1, chest_shift), desc=f"{subj_str}")
        
        for start in pbar:
            end = start + chest_pts
            
            window_labels = labels[start:end]
            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            
            if majority_label not in [1, 2, 3, 4]:
                skipped_count += 1
                pbar.set_postfix({'Saved': saved_count, 'Skipped (Label 0)': skipped_count})
                continue 

            start_sec = start / 700.0
            segment_dict = {'label': torch.tensor(majority_label, dtype=torch.long)}
            
            for mod, fs in freqs.items():
                # 🚨 Use our new mapping dictionary to pull the exact location and sensor name
                loc, sensor = WESAD_PKL_KEYS[mod]
                mod_start = int(start_sec * fs)
                mod_end = int((start_sec + window_sec) * fs)
                
                # Now this will not throw a KeyError!
                sig = signals[loc][sensor][mod_start:mod_end]
                
                if len(sig.shape) == 1:
                    sig = np.expand_dims(sig, axis=1)
                    
                segment_dict[mod] = torch.tensor(sig).float()

            torch.save(segment_dict, os.path.join(subj_out, f'{subj_str}_seg{saved_count}.pt'))
            saved_count += 1
            pbar.set_postfix({'Saved': saved_count, 'Skipped (Label 0)': skipped_count})

if __name__ == "__main__":
    process_wesad()