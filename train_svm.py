"""
train_clinicalBERT_fusion_svm_refined.py
Frozen ClinicalBERT ➜ PCA(text) ⊕ scaled numeric ⊕ one-hot cat ➜ SVM
Optuna search (refined around previous optimum) + diagnostics.
"""

# ───────── imports ─────────
import re, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna, warnings; warnings.filterwarnings("ignore")
from abbreviations import abbreviation_dict

# ───────── globals ─────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 64
EMB_BATCH   = 64
SEED        = 42
N_TRIALS    = 1000                      # refined search
torch.manual_seed(SEED); np.random.seed(SEED)

# ───────── helper: clean text ─────────
def clean_text(t: str) -> str:
    t = re.sub(r"[?,]", "", t)
    t = " ".join([abbreviation_dict.get(w, w) for w in t.split()]).lower()
    t = re.sub(r"\d+", "", t)
    return re.sub(r"[^\w\s]", "", t)

# ───────── load CSV ─────────
train_df = pd.read_csv("data/ktas_train.csv", on_bad_lines="skip")
val_df   = pd.read_csv("data/ktas_val.csv",   on_bad_lines="skip")
for df in (train_df, val_df):
    df.drop(columns=["Patients number per hour", "Diagnosis in ED"], inplace=True)
    df["Chief_complain"] = df["Chief_complain"].apply(clean_text)

cat_cols = ["Sex","Arrival mode","Injury","Mental","Pain"]
num_cols = ["Age","SBP","DBP","HR","RR","BT","Saturation","NRS_pain"]

# ───────── ClinicalBERT embedder ─────
model_name = "medicalai/ClinicalBERT"
tok  = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name).to(device).eval()
EMB_DIM = bert.config.hidden_size          # 768

@torch.no_grad()
def embed(texts):
    out = []
    for i in range(0, len(texts), EMB_BATCH):
        enc = tok(texts[i:i+EMB_BATCH], padding="max_length",
                  truncation=True, max_length=MAX_LEN, return_tensors="pt")
        vec = bert(enc["input_ids"].to(device),
                   attention_mask=enc["attention_mask"].to(device)
                  ).last_hidden_state.mean(1)
        out.append(vec.cpu().numpy())
    return np.vstack(out)

print("Extracting ClinicalBERT embeddings …")
X_txt_tr  = embed(train_df["Chief_complain"].tolist())
X_txt_val = embed(val_df["Chief_complain"].tolist())

# ───────── numeric / categorical ─────
scaler     = StandardScaler()
X_num_tr   = scaler.fit_transform(train_df[num_cols])
X_num_val  = scaler.transform(val_df[num_cols])

ohe        = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat_tr   = ohe.fit_transform(train_df[cat_cols])
X_cat_val  = ohe.transform(val_df[cat_cols])

# ───────── labels ─────────
le    = LabelEncoder()
y_tr  = le.fit_transform(train_df["KTAS_expert"])
y_val = le.transform(val_df["KTAS_expert"])

# legal PCA bound
MAX_PCA = min(X_txt_tr.shape[0] - 1, EMB_DIM)

# best vals from previous sweep
BEST = {
    "pca_dim": 560,
    "C": 114.4979,
    "kernel": "poly",
    "gamma": 2.8915e-3,
    "degree": 4,
    "coef0": 0.1695,
    "class_weight": "balanced"
}

# ───────── Optuna objective ─────────
def objective(trial):

    # --- PCA dim (±80 around 560, step 16) ---
    pca_low  = max(16, BEST["pca_dim"] - 80)
    pca_high = min(MAX_PCA, BEST["pca_dim"] + 80)
    dims = list(range(pca_low, pca_high+1, 16))
    pca_dim = trial.suggest_categorical("pca_dim", dims)

    # --- SVM params (log windows ±1 order) ---
    C      = trial.suggest_float("C", BEST["C"]/10, BEST["C"]*10, log=True)
    kernel = trial.suggest_categorical("kernel", ["poly","rbf","sigmoid"])
    if kernel == "linear":                              # never chosen here
        gamma, degree, coef0 = "scale", 3, 0.0
    else:
        gamma = trial.suggest_float("gamma", BEST["gamma"]/10,
                                    BEST["gamma"]*10, log=True)
        if kernel == "poly":
            degree = trial.suggest_int("degree",
                                       max(2, BEST["degree"]-1),
                                       min(5, BEST["degree"]+1))
            coef0  = trial.suggest_float("coef0",
                                         max(0.0, BEST["coef0"]-0.2),
                                         min(1.0, BEST["coef0"]+0.2))
        else:
            degree = 3
            coef0  = trial.suggest_float("coef0", 0.0, 0.5)

    cw = trial.suggest_categorical("class_weight", ["balanced", None])

    # --- PCA transform + fusion ---
    pca  = PCA(n_components=pca_dim, random_state=SEED)
    Z_tr = pca.fit_transform(X_txt_tr)
    Z_val= pca.transform(X_txt_val)

    X_tr_full  = np.hstack([Z_tr,  X_num_tr,  X_cat_tr])
    X_val_full = np.hstack([Z_val, X_num_val, X_cat_val])

    clf = SVC(C=C, kernel=kernel, gamma=gamma,
              degree=degree, coef0=coef0,
              class_weight=cw, random_state=SEED)

    clf.fit(X_tr_full, y_tr)
    acc = accuracy_score(y_val, clf.predict(X_val_full))

    trial.set_user_attr("clf", clf)
    trial.set_user_attr("pca", pca)
    return acc

# ───────── Optuna run ───────
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
# enqueue the previous best exactly once
study.enqueue_trial(BEST)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ───────── curves ───────────
vals  = [t.value for t in study.trials if t.value is not None]
best  = np.maximum.accumulate(vals)
tr_ix = np.arange(1, len(vals)+1)

plt.figure(figsize=(7,4))
plt.plot(tr_ix, vals, label="val accuracy")
plt.plot(tr_ix, best, color="red", label="cumulative best")
plt.xlabel("Trial"); plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Trial"); plt.legend(); plt.show()

plt.figure(figsize=(7,4))
plt.plot(tr_ix, 1-np.array(vals), label="val loss (1-acc)")
plt.plot(tr_ix, 1-best, color="red", label="best loss so far")
plt.xlabel("Trial"); plt.ylabel("Loss")
plt.title("Validation Loss per Trial"); plt.legend(); plt.show()

print("\n★★ New best val acc:", study.best_value)
print("★★ New best params :", study.best_params)

# ───────── evaluate best ─────
best_pca  = study.best_trial.user_attrs["pca"]
best_clf  = study.best_trial.user_attrs["clf"]

Z_val     = best_pca.transform(X_txt_val)
X_val_f   = np.hstack([Z_val, X_num_val, X_cat_val])
y_pred    = best_clf.predict(X_val_f)

print("\nClassification Report:")
print(classification_report(y_val, y_pred,
                            target_names=[str(c) for c in le.classes_]))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix"); plt.show()
