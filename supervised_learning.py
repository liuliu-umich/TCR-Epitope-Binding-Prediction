import argparse, json, numpy as np, pandas as pd, joblib
from typing import Optional, Sequence, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}

defaults = {
  "vdj_tcr": "cdr3",
  "vdj_epi": "antigen.epitope",
  "vdj_score": "vdjdb.score",
  "trait_tcr": "CDR3β",
  "trait_epi": "Epitope",
  "trait_bind": "Binding",
  "pos_score": 3,
  "neg_scores": (1, 2),
  "neg_ratio": 1.0,
  "maxlen_tcr": 30,
  "maxlen_epi": 15,
}

label_map = {
  'binder':1,'binding':1,'binds':1,'yes':1,'true':1,'positive':1,'pos':1,'1':1,
  'nonbinder':0,'non-binding':0,'nonbinding':0,'no':0,'false':0,'negative':0,'neg':0,'0':0
}

def load_any(path: str, sheet: Optional[str]=None) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"): return pd.read_csv(path)
    if p.endswith(".tsv"): return pd.read_csv(path, sep="\t")
    # Pandas returns a dict of DataFrames when sheet_name=None for Excel files.
    # Default to the first sheet so the CLI works with ordinary single-sheet workbooks.
    return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)

def clean_seq(s: str) -> Optional[str]:
    if pd.isna(s): return None
    s = "".join(ch for ch in str(s).upper() if ch.isalpha())
    s = "".join(ch for ch in s if ch in amino_acid_to_index)
    return s if len(s) >= 3 else None

def labels_to_binary(series: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x,(int,float,np.integer,np.floating)): return int(float(x)>0)
        return label_map.get(str(x).strip().lower(), np.nan)
    return series.apply(f)

def one_hot(seq: str, max_len: int) -> np.ndarray:
    m = np.zeros((max_len, len(amino_acids)), dtype=np.float32)
    L = min(len(seq), max_len)
    for i in range(L):
        m[i, amino_acid_to_index[seq[i]]] = 1.0
    return m

def featurize_df(X: pd.DataFrame, maxlen_tcr: int, maxlen_epi: int) -> np.ndarray:
    rows = []
    for tcr, epi in zip(X["tcr"], X["epitope"]):
        rows.append(np.concatenate([one_hot(tcr, maxlen_tcr).ravel(),
                                    one_hot(epi, maxlen_epi).ravel()]))
    return np.vstack(rows)

class FeatWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, maxlen_tcr:int, maxlen_epi:int):
        self.maxlen_tcr=maxlen_tcr
        self.maxlen_epi=maxlen_epi
    def fit(self, X, y=None): return self
    def transform(self, X):   return featurize_df(X, self.maxlen_tcr, self.maxlen_epi)

def build_vdj_supervised(df: pd.DataFrame, tcr_col: str, epi_col: str, score_col: str,
                         pos_score: int, neg_scores: Sequence[int], neg_ratio: float,
                         rng: np.random.RandomState) -> Tuple[pd.DataFrame, pd.Series]:
    df = df[[tcr_col, epi_col, score_col]].copy()
    df[tcr_col] = df[tcr_col].map(clean_seq)
    df[epi_col] = df[epi_col].map(clean_seq)
    df = df.dropna(subset=[tcr_col, epi_col]).drop_duplicates().reset_index(drop=True)
    pos = df[df[score_col]==pos_score][[tcr_col, epi_col]].drop_duplicates().reset_index(drop=True)
    if pos.empty:
        raise SystemExit(f"No positives (score=={pos_score}) in VDJdb.")
    pos["label"] = 1
    neg_pool = df[df[score_col].isin(neg_scores)][[tcr_col, epi_col]].drop_duplicates().reset_index(drop=True)
    if neg_pool.empty:
        raise SystemExit(f"No negatives pool (score in {set(neg_scores)}) in VDJdb.")
    n_neg = int(np.ceil(len(pos) * neg_ratio))
    neg_pool = neg_pool.sample(n_neg, replace=len(neg_pool)<n_neg, random_state=rng.randint(1,1_000_000))
    shuffled_epi = neg_pool[epi_col].sample(frac=1.0, random_state=rng.randint(1,1_000_000)).reset_index(drop=True)
    neg = neg_pool[[tcr_col]].copy().reset_index(drop=True)
    neg[epi_col] = shuffled_epi
    same = neg[epi_col].values == neg_pool[epi_col].values
    if same.any():
        idx = np.where(same)[0]
        neg.iloc[idx, neg.columns.get_loc(epi_col)] = np.roll(neg.iloc[idx, neg.columns.get_loc(epi_col)].values, 1)
    neg["label"] = 0
    pairs = pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=rng.randint(1,1_000_000))
    X = pairs.rename(columns={tcr_col:"tcr", epi_col:"epitope"})[["tcr","epitope"]]
    y = pairs["label"].astype(int)
    return X, y

def build_trait_validation(df: pd.DataFrame, tcr_col: str, epi_col: str,
                           bind_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    df[tcr_col] = df[tcr_col].map(clean_seq)
    df[epi_col] = df[epi_col].map(clean_seq)
    df = df.dropna(subset=[tcr_col, epi_col]).drop_duplicates().reset_index(drop=True)
    X = df.rename(columns={tcr_col:"tcr", epi_col:"epitope"})[["tcr","epitope"]]
    y = None
    if bind_col and bind_col in df.columns:
        y_try = labels_to_binary(df[bind_col])
        if y_try.notna().any():
            keep = y_try.notna()
            X = X.loc[keep].reset_index(drop=True)
            y = y_try.loc[keep].astype(int).reset_index(drop=True)
    return X, y

def train_vdjdb(vdj_path: str, sheet: Optional[str], tcr_col: str, epi_col: str, score_col: str,
                pos_score: int, neg_scores: Sequence[int], neg_ratio: float,
                maxlen_tcr:int, maxlen_epi:int, model_out: str):
    rng = np.random.RandomState(42)
    df = load_any(vdj_path, sheet)
    X, y = build_vdj_supervised(df, tcr_col, epi_col, score_col, pos_score, neg_scores, neg_ratio, rng)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("fe", FeatWrapper(maxlen_tcr, maxlen_epi)),
        ("sc", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                   max_iter=2000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["roc_auc","f1","accuracy","precision","recall"])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:,1]
    yhat = (p>=0.5).astype(int)
    metrics = {
        "cv_mean": {k.replace("test_",""): float(np.mean(v)) for k,v in cv_scores.items() if k.startswith("test_")},
        "cv_std":  {k.replace("test_",""): float(np.std(v))  for k,v in cv_scores.items() if k.startswith("test_")},
        "test": {
            "roc_auc": float(roc_auc_score(yte, p)),
            "f1": float(f1_score(yte, yhat)),
            "accuracy": float(accuracy_score(yte, yhat)),
            "precision": float(precision_score(yte, yhat)),
            "recall": float(recall_score(yte, yhat)),
        },
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "repr":"onehot_lr",
        "maxlen_tcr":maxlen_tcr,
        "maxlen_epi":maxlen_epi,
        "tcr_col":"tcr",
        "epitope_col":"epitope",
        "score_col":score_col,
        "pos_score":pos_score,
        "neg_scores":list(neg_scores),
        "neg_ratio":neg_ratio
    }
    joblib.dump({"pipeline": pipe, "tcr_col":"tcr", "epitope_col":"epitope", "metrics": metrics}, model_out)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to: {model_out}")

def validate_trait(model_path: str, trait_path: str, sheet: Optional[str],
                   tcr_col: str, epi_col: str, bind_col: Optional[str]):
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    df = load_any(trait_path, sheet)
    X, y = build_trait_validation(df, tcr_col, epi_col, bind_col)
    p = pipe.predict_proba(X)[:,1]
    yhat = (p>=0.5).astype(int)
    if y is not None:
        metrics = {
            "external": {
                "roc_auc": float(roc_auc_score(y, p)),
                "f1": float(f1_score(y, yhat)),
                "accuracy": float(accuracy_score(y, yhat)),
                "precision": float(precision_score(y, yhat)),
                "recall": float(recall_score(y, yhat)),
            },
            "n_val": int(len(X)),
            "tcr_col":"tcr",
            "epitope_col":"epitope",
            "label_col":bind_col
        }
    else:
        metrics = {
            "note":"No usable labels in TRAITdb dataset, binding might be missing or not mappable. Reporting score summary.",
            "n_val": int(len(X)),
            "summary": {
                "mean_prob": float(np.mean(p)),
                "std_prob": float(np.std(p)),
                "pct_pred_positive@0.5": float(np.mean(yhat))
            }
        }
    print(json.dumps(metrics, indent=2))

def main():
    ap = argparse.ArgumentParser(description="TCR–Epitope classifier: train on VDJdb, validate on TRAITdb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="Train on VDJdb")
    tr.add_argument("--vdjdb_path", required=True)
    tr.add_argument("--vdjdb_sheet", default=None)
    tr.add_argument("--tcr_col", default=defaults["vdj_tcr"])
    tr.add_argument("--epitope_col", default=defaults["vdj_epi"])
    tr.add_argument("--score_col", default=defaults["vdj_score"])
    tr.add_argument("--pos_score", type=int, default=defaults["pos_score"])
    tr.add_argument("--neg_scores", type=int, nargs="+", default=list(defaults["neg_scores"]))
    tr.add_argument("--neg_ratio", type=float, default=defaults["neg_ratio"])
    tr.add_argument("--maxlen_tcr", type=int, default=defaults["maxlen_tcr"])
    tr.add_argument("--maxlen_epi", type=int, default=defaults["maxlen_epi"])
    tr.add_argument("--model_out", default="tcr_epitope_onehot_lr.pkl")
    va = sub.add_parser("validate", help="External validation on TRAITdb")
    va.add_argument("--model", required=True)
    va.add_argument("--trait_path", required=True)
    va.add_argument("--trait_sheet", default=None)
    va.add_argument("--tcr_col", default=defaults["trait_tcr"])
    va.add_argument("--epitope_col", default=defaults["trait_epi"])
    va.add_argument("--bind_col", default=defaults["trait_bind"])
    args = ap.parse_args()
    if args.cmd == "train":
        train_vdjdb(args.vdjdb_path, args.vdjdb_sheet, args.tcr_col, args.epitope_col, args.score_col,
                    args.pos_score, tuple(args.neg_scores), args.neg_ratio,
                    args.maxlen_tcr, args.maxlen_epi, args.model_out)
    else:
        validate_trait(args.model, args.trait_path, args.trait_sheet, args.tcr_col, args.epitope_col, args.bind_col)

if __name__ == "__main__":
    main()
