from pathlib import Path
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd

COHORTS = {
    "core": ("data/core_3539_metadata.csv", "work/features", 3539),
    "main": ("data/main_10694_metadata.csv", "full_cohort/features", 10694),
    "lf": ("data/aux_lf87_metadata.csv", "aux_lf87/features", 87),
}

def _read(name, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    if root.is_file():
        with zipfile.ZipFile(root) as z:
            return z.read(name)
    return (root / name).read_bytes()

def load_metadata(cohort="main", root=None):
    path, _, n = COHORTS[cohort]
    df = pd.read_csv(BytesIO(_read(path, root)), dtype={"cut_id": str},
                     keep_default_na=False, low_memory=False)
    if len(df) != n or not df.cut_id.is_unique:
        raise ValueError("Unexpected cohort metadata")
    return df

def load_feature(feature="vggish", cohort="main", root=None):
    df = load_metadata(cohort, root)
    _, folder, n = COHORTS[cohort]
    with np.load(BytesIO(_read(folder + "/" + feature + ".npz", root)),
                 allow_pickle=False) as z:
        ids = z["cut_id"].astype(str)
        X, names = z["X"].copy(), z["names"].copy()
    if not np.array_equal(ids, df.cut_id.to_numpy(dtype=str)) or len(X) != n:
        raise ValueError("Feature/metadata IDs do not match")
    return df, X, names

def load_inventory(root=None):
    return pd.read_csv(BytesIO(_read("data/bench_manifest.csv", root)),
                       dtype={"cut_id": str}, keep_default_na=False, low_memory=False)

def load_cached_umap3d(feature="vggish", root=None):
    df = load_metadata("main", root)
    with np.load(BytesIO(_read("cached_umap3d/" + feature + ".npz", root)),
                 allow_pickle=False) as z:
        if not np.array_equal(z["cut_id"].astype(str), df.cut_id.to_numpy(dtype=str)):
            raise ValueError("Coordinates/metadata IDs do not match")
        return df, z["coords"].copy(), z["cluster8"].copy()

# Examples after extracting:
# metadata, X, feature_names = load_feature("vggish", "main")
# metadata, X, feature_names = load_feature("ast_tile", "core")
# For direct ZIP access: load_feature("physv0", "main", "Watkins_Claude_Input_Package.zip")
