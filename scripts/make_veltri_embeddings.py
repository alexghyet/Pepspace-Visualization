from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

#paths
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import parse_amp_labels

CSV_PATH = BASE / "data" / "veltri" / "all_veltri.csv"

OUT_DIR = BASE / "embeddings" / "veltri"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "veltri_embeddings.npy"
META_PATH = OUT_DIR / "veltri_metadata.csv"

#ProteoGPT model
MODEL_ID = "nferruz/ProtGPT2"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV_PATH)

#actual column names
SEQ_COL = "aa_seq"   # peptide strings
LABEL_COL = "AMP"    # 1/0 label

missing_columns = {SEQ_COL, LABEL_COL}.difference(df.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}.")
sequence_values = df[SEQ_COL]
if df.empty or sequence_values.isna().any() or sequence_values.astype(str).str.strip().eq("").any():
    raise ValueError("Every dataset row must contain a non-empty peptide sequence.")
sequences = sequence_values.astype(str).str.strip().tolist()

# Reject missing or unknown classes instead of treating them as Non-AMP.
is_amp = parse_amp_labels(df[LABEL_COL])

# Load tokenizer + model
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# no pad token by default
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

#mean pooling; ignore padding
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B,1)
    return summed / counts


# Embed sequences
batch_size = 16
vecs = []

with torch.no_grad():
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]

        tok = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        out = model(**tok)  # out.last_hidden_state: (B,T,H)
        pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])  # (B,H)

        vecs.append(pooled.cpu().numpy().astype("float32"))
        print(f"Embedded {min(i+batch_size, len(sequences))}/{len(sequences)}")

X = np.vstack(vecs)


# Save outputs
np.save(EMB_PATH, X)
pd.DataFrame({
    "sequence": sequences,
    "is_amp": is_amp.astype(bool)
}).to_csv(META_PATH, index=False)

print("Saved embeddings:", EMB_PATH, X.shape)
print("Saved metadata:", META_PATH)
