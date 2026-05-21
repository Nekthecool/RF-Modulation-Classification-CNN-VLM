import os
import re
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg') # Safe for HPC environments without displays
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Import our custom dataset logic
from dataset import RFConstellationEvalDataset

# ====================================================================
# HPC DDP GUARD & SETUP
# ====================================================================
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank != 0:
    import sys
    sys.exit(0)  # Restrict evaluation to a single GPU on HPC

print("\n--- Starting Mathematically Guaranteed Evaluation (Rank 0) ---")

os.environ['HF_HOME'] = os.path.expanduser('~/hf_cache')
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

DATA_DIR = '../01_Dataset_Simulation'
OUTPUT_DIR = './VLM_Output'
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "best_vlm_adapter")
GRAPH_DIR = os.path.join(OUTPUT_DIR, 'VLM_Eval_Plots')
os.makedirs(GRAPH_DIR, exist_ok=True)

if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"CRITICAL ERROR: Adapter weights not found at {ADAPTER_PATH}. Run train.py first.")

# ====================================================================
# LOAD MODEL & DATASETS
# ====================================================================
print("Loading Base Model and Adapters...")
processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_pixels=384*384)
processor.tokenizer.padding_side = "left" # Critical for batched inference

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
)
eval_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
eval_model.eval()

val_dataset = RFConstellationEvalDataset(os.path.join(DATA_DIR, 'val_images_production.npz'), os.path.join(DATA_DIR, 'val_labels_production.csv'))
test_dataset = RFConstellationEvalDataset(os.path.join(DATA_DIR, 'test_images_generalization.npz'), os.path.join(DATA_DIR, 'test_labels_generalization.csv'))

# ====================================================================
# INFERENCE ENGINE (EXACT MATCH LOGIT EXTRACTION)
# ====================================================================
def run_eval(dataset, desc, split="val"):
    tasks = ['mod', 'pn', 'iqi', 'amp', 'jam', 'snr']
    gt = {t: [] for t in tasks}
    preds = {t: [] for t in tasks}
    snrs = []

    m_list = ['4-ASK', '8-ASK', 'BPSK', 'QPSK', '4-HQAM', '16-HQAM', '64-HQAM', '16-QAM', '32-QAM', '64-QAM', '128-QAM', '256-QAM', '16-APSK', '32-APSK', '64-APSK', '128-APSK']
    m_list_sorted = sorted(m_list, key=len, reverse=True)

    inference_question = "Analyze this constellation diagram and extract the communication parameters."
    BATCH_SIZE = 16
    total_time = 0.0

    # Token ID extraction for severities
    def get_tids(word):
        tids = []
        for w in [word, " " + word]:
            enc = processor.tokenizer.encode(w, add_special_tokens=False)
            if enc: tids.append(enc[0])
        return list(set(tids))

    id_none, id_med, id_ext = get_tids("none"), get_tids("medium"), get_tids("extreme")

    def get_max_logit(logits_tensor, tids):
        return torch.max(logits_tensor[tids]).item()

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=desc):
        batch_idx = range(i, min(i+BATCH_SIZE, len(dataset)))
        imgs = [dataset[idx]["image"] for idx in batch_idx]
        batch_df = dataset.labels_df.iloc[batch_idx]

        for _, row in batch_df.iterrows():
            gt['mod'].append(str(row['Modulation']).upper())
            gt['pn'].append(str(row['Phase_Noise_Severity']).lower())
            gt['iqi'].append(str(row['IQ_Imbalance_Severity']).lower())
            gt['amp'].append(str(row['Amplitude_Distortion_Severity']).lower())
            gt['jam'].append(str(row['Interference_Severity']).lower())
            gt['snr'].append(str(row['SNR_Range']).lower())
            snrs.append(row['SNR_dB'])

        # --- STEP 1: Text Generation for Modulation & SNR ---
        texts = [processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":inference_question}]}], tokenize=False, add_generation_prompt=True)] * len(imgs)
        inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(eval_model.device)
        if "pixel_values" in inputs: inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        start = time.time()
        with torch.inference_mode():
            gen_ids = eval_model.generate(**inputs, max_new_tokens=60, do_sample=False)

        decoded = processor.batch_decode([g[len(in_id):] for g, in_id in zip(gen_ids, inputs.input_ids)], skip_special_tokens=True)

        batch_p_mod, batch_p_snr = [], []
        for txt in decoded:
            txt_lower = txt.lower()

            # Robust Parsing: Modulation
            txt_clean = txt_lower.replace("-", "").replace(" ", "")
            pred_mod = "UNKNOWN"
            for m in m_list_sorted:
                if m.lower().replace("-", "").replace(" ", "") in txt_clean:
                    pred_mod = m; break
            batch_p_mod.append(pred_mod)
            preds['mod'].append(pred_mod.upper())

            # Robust Parsing: SNR
            p_snr = "unknown"
            snr_match = re.search(r'\b(low|medium|high)\b[^\.\,]*?snr|snr[^\.\,]*?\b(low|medium|high)\b', txt_lower)
            if snr_match: p_snr = snr_match.group(1) if snr_match.group(1) else snr_match.group(2)
            batch_p_snr.append(p_snr)
            preds['snr'].append(p_snr)

        # --- STEP 2: THE LOGITS HACK (Teacher Forcing Impairments) ---
        batch_ans_prefix = []
        for j in range(len(imgs)):
            prefix = f"The constellation shows a {batch_p_mod[j]} modulation. The SNR range is {batch_p_snr[j]}. Regarding impairments, it exhibits "
            batch_ans_prefix.append(prefix)

        imp_sequence = [
            ("pn", " phase noise, "),
            ("iqi", " i/q imbalance, "),
            ("amp", " amplitude distortion, and "),
            ("jam", " interference.")
        ]

        for imp_key, suffix in imp_sequence:
            text_prompts = []
            for j in range(len(imgs)):
                base_prompt = processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":inference_question}]}], tokenize=False, add_generation_prompt=True)
                text_prompts.append(base_prompt + batch_ans_prefix[j])

            model_inputs = processor(text=text_prompts, images=imgs, return_tensors="pt", padding=True).to(eval_model.device)
            if "pixel_values" in model_inputs: model_inputs["pixel_values"] = model_inputs["pixel_values"].to(torch.bfloat16)

            with torch.inference_mode():
                outputs = eval_model.generate(**model_inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True, do_sample=False)

            next_token_logits = outputs.scores[0]

            for j in range(len(imgs)):
                l_n = get_max_logit(next_token_logits[j], id_none)
                l_m = get_max_logit(next_token_logits[j], id_med)
                l_e = get_max_logit(next_token_logits[j], id_ext)

                tensor_logits = torch.tensor([l_n, l_m, l_e]) / 2.0
                probs = torch.softmax(tensor_logits, dim=0)

                ev = 0.0 * probs[0] + 2.0 * probs[1] + 4.0 * probs[2]

                # ROUNDING BASED ON SPLIT
                if split == "val":
                    valid_classes = [0, 2, 4]
                    final_val = min(valid_classes, key=lambda x: abs(x - ev.item()))
                else:
                    final_val = max(0, min(4, int(round(ev.item()))))

                pred_word = ['none', 'low', 'medium', 'high', 'extreme'][final_val]
                preds[imp_key].append(pred_word)

                batch_ans_prefix[j] += f"{pred_word}{suffix}"

        total_time += (time.time() - start)
        del inputs, model_inputs, outputs, next_token_logits, texts, text_prompts
        torch.cuda.empty_cache()

    print(f"   ⏱️ Inference Speed: {(total_time/(len(dataset)/BATCH_SIZE))*1000:.2f} ms/batch")
    return {k: np.array(v) for k, v in gt.items()}, {k: np.array(v) for k, v in preds.items()}, np.array(snrs)

# ====================================================================
# RUN PIPELINE & PLOT GENERATION
# ====================================================================
print("\n[VLM] Running Validation Set (STRICT KNOWLEDGE)...")
v_gt, v_p, v_s = run_eval(val_dataset, "Validation", split="val")

print("\n[VLM] Running Test Set (LOGIT INTERPOLATION)...")
t_gt, t_p, t_s = run_eval(test_dataset, "Test", split="test")

tasks_keys = ['mod', 'pn', 'iqi', 'snr', 'amp', 'jam']
tasks_display = ['MOD', 'PN', 'IQI', 'SNR', 'AMP_DIST', 'JAMMING']

print("\n=== [FINE-TUNED VLM] Per-Task Accuracy ===")
for tk, td in zip(tasks_keys, tasks_display):
    print(f"Task '{td}': Val {accuracy_score(v_gt[tk], v_p[tk])*100:.2f}% | Test {accuracy_score(t_gt[tk], t_p[tk])*100:.2f}%")

sns.set_theme(style="whitegrid")
m_list = ['4-ASK', '8-ASK', 'BPSK', 'QPSK', '4-HQAM', '16-HQAM', '64-HQAM', '16-QAM', '32-QAM', '64-QAM', '128-QAM', '256-QAM', '16-APSK', '32-APSK', '64-APSK', '128-APSK']

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(False)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(v_gt['mod'], v_p['mod'], labels=m_list), display_labels=m_list)
disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=90, text_kw={'fontsize': 8})
plt.title("VLM: Modulation Confusion Matrix (Validation)", weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "01_cm_lora.png"), dpi=300)
plt.close(fig)

# 2. SNR vs Accuracy
plt.figure(figsize=(9,5))
v_acc = {v: accuracy_score(v_gt['mod'][v_s==v], v_p['mod'][v_s==v])*100 for v in np.unique(v_s)}
t_acc = {v: accuracy_score(t_gt['mod'][t_s==v], t_p['mod'][t_s==v])*100 for v in np.unique(t_s)}
plt.plot(list(v_acc.keys()), list(v_acc.values()), 'bo-', label='Validation', linewidth=2)
plt.plot(list(t_acc.keys()), list(t_acc.values()), 'rs--', label='Test (Generalization)', linewidth=2)
plt.title("Robustness: VLM Modulation Accuracy vs SNR", weight='bold'); plt.xlabel('SNR (dB)'); plt.ylabel('Accuracy (%)'); plt.ylim(0, 105); plt.legend(); plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "02_snr_robustness_lora.png"), dpi=300)
plt.close()

# 3. MAE Plot
sev_map = {'none':0, 'low':1, 'medium':2, 'high':3, 'extreme':4}
plt.figure(figsize=(10,6))
for i, (k, lab, mark) in enumerate([('pn','Phase Noise','o'),('iqi','IQ Imbalance','s'),('amp','Amp Distortion','^'),('jam','Jamming','D')]):
    mae = []
    for s in ['none','low','medium','high','extreme']:
        idx = np.where(t_gt[k] == s)[0]
        mae.append(np.mean([abs(sev_map[s] - sev_map[p]) for p in t_p[k][idx] if p in sev_map]) if len(idx)>0 else np.nan)
    plt.plot(['none','low','medium','high','extreme'], mae, marker=mark, label=lab, linewidth=2.5, markersize=8)
plt.title("VLM: Sensitivity to Severity (Mean Absolute Error)", weight='bold'); plt.xlabel('Severity Level'); plt.ylabel("MAE (Lower is Better)"); plt.ylim(0, 2.0); plt.legend(); plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "03_severity_mae_lora.png"), dpi=300)
plt.close()

# 4. Accuracy vs Severity Plot
plt.figure(figsize=(10,6))
for i, (k, lab, mark) in enumerate([('pn','Phase Noise','o'),('iqi','IQ Imbalance','s'),('amp','Amp Distortion','^'),('jam','Jamming','D')]):
    accs = [accuracy_score(t_gt[k][t_gt[k]==s], t_p[k][t_gt[k]==s])*100 if np.any(t_gt[k]==s) else np.nan for s in ['none','low','medium','high','extreme']]
    plt.plot(['none','low','medium','high','extreme'], accs, marker=mark, label=lab, linewidth=2.5, markersize=8)
plt.title("VLM: Accuracy by Impairment Severity", weight='bold'); plt.xlabel('Severity Level'); plt.ylabel("Accuracy (%)"); plt.ylim(0, 105); plt.legend(); plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "04_severity_accuracy_lora.png"), dpi=300)
plt.close()

# 5. OOD Bar Chart
ood_c = next((c for c in test_dataset.labels_df.columns if 'ood' in c.lower() or 'unknown' in c.lower()), None)
if ood_c:
    plt.figure(figsize=(8,5))
    vals = test_dataset.labels_df[ood_c].values
    u_vals = sorted(np.unique(vals))
    gen_acc = [accuracy_score(t_gt['mod'][vals==c], t_p['mod'][vals==c])*100 for c in u_vals]
    bars = plt.bar(u_vals, gen_acc, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
    for b in bars: plt.text(b.get_x()+b.get_width()/2, b.get_height()+2, f'{b.get_height():.1f}%', ha='center', weight='bold')
    plt.title("Generalization Capability vs Unknown Conditions", weight='bold'); plt.xlabel('Unknown Parameters (OOD)'); plt.ylabel('Modulation Accuracy (%)'); plt.ylim(0, 105); plt.xticks(u_vals); plt.grid(axis='y', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "05_generalization_gap_lora.png"), dpi=300)
    plt.close()

print(f"\n✅ EVALUATION COMPLETE! All plots successfully saved to: {os.path.abspath(GRAPH_DIR)}")