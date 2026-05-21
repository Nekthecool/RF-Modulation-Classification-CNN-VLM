import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import zipfile

# ====================================================================
# CUSTOM PYTORCH DATASET FOR CONSTELLATION IMAGES
# ====================================================================
class ConstellationDataset(Dataset):
    def __init__(self, images_path, labels_path, temp_npy_name):
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Archive not found at {images_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Labels not found at {labels_path}")

        # Local extraction directory for memory-mapped arrays
        temp_dir = './temp_unpacked_data'
        os.makedirs(temp_dir, exist_ok=True)
        
        extracted_file_path = os.path.join(temp_dir, temp_npy_name)

        # Extract only if it hasn't been extracted yet
        if not os.path.exists(extracted_file_path):
            print(f"Extracting {os.path.basename(images_path)} to memory-mapped local storage...")
            with zipfile.ZipFile(images_path, 'r') as zf:
                zf.extract('images.npy', path=temp_dir)
                os.rename(os.path.join(temp_dir, 'images.npy'), extracted_file_path)

        # Load using memory mapping to prevent RAM overflow
        self.images = np.load(extracted_file_path, mmap_mode='r')
        self.df = pd.read_csv(labels_path)

        # Encode categorical variables
        self.all_mods = [
            '4-ASK', '8-ASK', 'BPSK', 'QPSK', '4-HQAM', '16-HQAM',
            '64-HQAM', '16-QAM', '32-QAM', '64-QAM', '128-QAM',
            '256-QAM', '16-APSK', '32-APSK', '64-APSK', '128-APSK'
        ]
        self.le_mod = LabelEncoder().fit(self.all_mods)
        self.labels_mod = self.le_mod.transform(self.df['Modulation'])

        severity_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'extreme': 4}
        self.labels_pn = self.df['Phase_Noise_Severity'].str.lower().map(severity_map).values
        self.labels_iqi = self.df['IQ_Imbalance_Severity'].str.lower().map(severity_map).values
        self.labels_amp_dist = self.df['Amplitude_Distortion_Severity'].str.lower().map(severity_map).values
        self.labels_snr = self.df['SNR_Range'].str.lower().map(severity_map).values
        self.labels_jamming = self.df['Interference_Severity'].str.lower().map(severity_map).values
        self.actual_snr_db = self.df['SNR_dB'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Bit unpacking and normalization for CNN consumption
        packed_img = self.images[idx]
        img = np.unpackbits(packed_img).reshape(384, 384).astype(np.float32)
        img_tensor = torch.tensor(img).unsqueeze(0) # Add channel dimension

        labels = {
            'mod': torch.tensor(self.labels_mod[idx], dtype=torch.long),
            'pn': torch.tensor(self.labels_pn[idx], dtype=torch.long),
            'iqi': torch.tensor(self.labels_iqi[idx], dtype=torch.long),
            'snr': torch.tensor(self.labels_snr[idx], dtype=torch.long),
            'amp_dist': torch.tensor(self.labels_amp_dist[idx], dtype=torch.long),
            'jamming': torch.tensor(self.labels_jamming[idx], dtype=torch.long)
        }
        return img_tensor, labels, self.actual_snr_db[idx]