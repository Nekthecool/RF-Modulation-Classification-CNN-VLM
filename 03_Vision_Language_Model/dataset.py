import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random

# ====================================================================
# VLM TRAINING DATASET (MULTI-TASK PROMPTING)
# ====================================================================
class RFConstellationTrainDataset(Dataset):
    def __init__(self, npz_file, csv_file, split="train"):
        self.images = np.load(npz_file, mmap_mode='r')['images']
        self.labels_df = pd.read_csv(csv_file)
        self.split = split

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Unpack bits from 1D array back to 384x384 grayscale [0, 255]
        packed_img = self.images[idx]
        gray_array = (np.unpackbits(packed_img).reshape(384, 384) * 255).astype(np.uint8)
        image = Image.fromarray(gray_array).convert("RGB")

        # Extract all labels required by the project specifications
        row = self.labels_df.iloc[idx]
        mod = row['Modulation']
        pn = row['Phase_Noise_Severity']
        iqi = row['IQ_Imbalance_Severity']
        amp = row['Amplitude_Distortion_Severity']
        jam = row['Interference_Severity']
        snr_range = row['SNR_Range']

        prompts = [
            "Analyze this constellation diagram and extract the communication parameters.",
            "Identify the modulation type, impairments, and SNR range from this RF signal representation.",
            "Perform a technical visual analysis of this RF constellation diagram."
        ]

        question_text = random.choice(prompts) if self.split == "train" else prompts[0]

        answer_text = (
            f"The constellation shows a {mod} modulation. "
            f"The SNR range is {snr_range}. "
            f"Regarding impairments, it exhibits {pn} phase noise, {iqi} i/q imbalance, "
            f"{amp} amplitude distortion, and {jam} interference."
        )

        # Build the message history structure required by Qwen
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        ]
        return {"image": image, "messages": messages}

# ====================================================================
# VLM EVALUATION DATASET (RAW IMAGE PROVISIONING)
# ====================================================================
class RFConstellationEvalDataset(Dataset):
    def __init__(self, npz_file, csv_file):
        self.images = np.load(npz_file, mmap_mode='r')['images']
        self.labels_df = pd.read_csv(csv_file)

    def __len__(self): 
        return len(self.labels_df)

    def decode_image(self, idx):
        packed_img = self.images[idx]
        gray_array = (np.unpackbits(packed_img).reshape(384, 384) * 255).astype(np.uint8)
        return Image.fromarray(gray_array).convert("RGB")

    def __getitem__(self, idx):
        return {"image": self.decode_image(idx), "row_data": self.labels_df.iloc[idx]}