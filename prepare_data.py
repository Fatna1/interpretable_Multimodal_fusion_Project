# Data audit for SNP + sMRI + fMRI (ADNI). Inspection stage.

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

MRI_MAT_PATH = "data/ADNIFullDataset.mat"
SNP_PATH = "data/labels.csv"
META1_PATH = "data/merge1.csv"
META2_PATH = "data/merge2.csv"
OUT_DIR = "outputs"
PLOT_DIR = os.path.join(OUT_DIR, "plots")

ID_COL = "subject_id"
DX_COL = "diagnosis"
AGE_COL = "age"
SEX_COL = "sex"

# Column names that may hold the subject ID in your files (ADNI often uses PTID or RID)
ID_CANDIDATES = ["subject_id", "subject", "ptid", "subjectid", "subj_id", "id", "rid"]
DX_CANDIDATES = ["diagnosis", "dx", "group", "dx_bl", "research group", "label"]
AGE_CANDIDATES = ["age", "age_bl", "pt_age"]
SEX_CANDIDATES = ["sex", "gender", "pt_sex", "ptgender"]

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def standardize_columns(df, candidates, target):
    """Rename the first matching column (case-insensitive) to the target name."""
    lookup = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return df.rename(columns={lookup[cand.lower()]: target})
    return df


def standardize_table(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = standardize_columns(df, ID_CANDIDATES, ID_COL)
    df = standardize_columns(df, DX_CANDIDATES, DX_COL)
    df = standardize_columns(df, AGE_CANDIDATES, AGE_COL)
    df = standardize_columns(df, SEX_CANDIDATES, SEX_COL)
    return df


def inspect_df(df, name):
    banner(f"TABLE: {name}")
    print("shape:", df.shape)
    print("columns:", list(df.columns)[:30], "..." if df.shape[1] > 30 else "")
    print("\ndtypes:\n", df.dtypes.value_counts().to_string())
    print("\nfirst rows:")
    print(df.iloc[:5, :8].to_string())
    return df


# inspect the MAT file

def inspect_mat(path):
    banner(f"MAT FILE: {path}")
    try:
        for name, shape, dtype in whosmat(path):
            print(f"  {name:25s} shape={shape}  dtype={dtype}")
    except Exception as e:
        print("whosmat failed:", e)

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        mat = {k: v for k, v in mat.items() if not k.startswith("__")}
    except NotImplementedError:
        import h5py
        mat = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                mat[k] = np.array(f[k])
        print("(loaded with h5py: MATLAB v7.3 file)")

    for k, v in mat.items():
        print(f"\n  variable '{k}': type={type(v).__name__}", end="")
        if isinstance(v, np.ndarray):
            print(f", shape={v.shape}, dtype={v.dtype}")
            if v.dtype.names:
                print("  struct fields:", v.dtype.names)
        else:
            print()
            if hasattr(v, "_fieldnames"):
                print("  struct fields:", v._fieldnames)
    return mat


# load the CSV files

def load_csv(path, label):
    df = pd.read_csv(path)
    inspect_df(df, label)
    return df


def load_metadata():
    """Load both metadata files and merge them on the subject ID."""
    m1 = standardize_table(load_csv(META1_PATH, "merge1.csv"))
    m2 = standardize_table(load_csv(META2_PATH, "merge2.csv"))

    print("\nmerge1 columns after standardizing:", list(m1.columns))
    print("merge2 columns after standardizing:", list(m2.columns))

    if ID_COL not in m1.columns or ID_COL not in m2.columns:
        raise ValueError("No subject ID column found. Add your column name to ID_CANDIDATES.")

    # Take only the columns from merge2 that merge1 does not already have
    extra = [c for c in m2.columns if c not in m1.columns or c == ID_COL]
    meta = m1.merge(m2[extra], on=ID_COL, how="outer")
    print("\nMerged metadata shape:", meta.shape)
    return meta


# inspection stage

if __name__ == "__main__":
    mat = inspect_mat(MRI_MAT_PATH)

    snp_df = inspect_df(load_csv(SNP_PATH, "labels.csv"), "SNP table")
    meta_df = inspect_df(load_metadata(), "Merged metadata")

    print("\nDiagnosis value counts:")
    if DX_COL in meta_df.columns:
        print(meta_df[DX_COL].value_counts(dropna=False).to_string())
    else:
        print("No diagnosis column found. Check the column names printed above.")

    if ID_COL in meta_df.columns:
        print("\nUnique subjects in metadata:", meta_df[ID_COL].nunique())
