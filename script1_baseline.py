"""
Script 1 – Baseline: Random Forest with repeated hold-out 
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")

DATASETS = {
    "neuroblastoma": {
        "path": "10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
        "target": "outcome",
        "drop": [],
    },
    "pediatric_brain_tumor": {
        "path": "dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
        "target": "Status OS",
        "drop": ["OS"],          
    },
    "colorectal_cancer": {
        "path": "dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
        "target": "death",
        "drop": ["bian_ma", "interval", "interval_r", "progress"],
    },
    "sepsis": {
        "path": "journal_pone_0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
        "target": "Mortality",
        "drop": ["LOS-ICU"],
    },
    "depression_heart_failure": {
        "path": "journal_pone_0158570_S2File_depression_heart_failure.csv",
        "target": "Death (1=Yes, 0=No)",
        "drop": [
            "Time from HF to Death (days)",
            "Time from HF to hospitalization (days)",
            "Hospitalized (1=Yes, 0=No)",
        ],
    },
}

N_ITER       = 100
TEST_SIZE    = 0.30
RANDOM_SEED  = 42
DATA_DIR     = os.path.dirname(os.path.abspath(__file__))


def load_dataset(name: str, data_dir: str):
    cfg    = DATASETS[name]
    path   = os.path.join(data_dir, cfg["path"])
    df     = pd.read_csv(path)
    df.columns = df.columns.str.strip()          
    target = cfg["target"].strip()
    drop   = [c.strip() for c in cfg["drop"]]

    X = df.drop(columns=[target] + drop, errors="ignore")
    y = df[target]

    # Impute remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.mode()[0])

    y_arr = y.values.astype(int)
    unique_labels = np.unique(y_arr)
    if not (set(unique_labels) <= {0, 1}):
        label_map = {v: i for i, v in enumerate(sorted(unique_labels))}
        y_arr = np.array([label_map[v] for v in y_arr])

    return X.values.astype(np.float32), y_arr


def repeated_holdout(X, y, n_iter=N_ITER):
    """100-iteration repeated hold-out, returns list of MCC scores."""
    clf    = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    scaler = StandardScaler()
    mccs   = []

    for i in range(n_iter):
        seed = RANDOM_SEED + i
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf.set_params(random_state=seed)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        mccs.append(matthews_corrcoef(y_te, preds))

    return mccs


def main(data_dir: str = DATA_DIR):
    results = {}

    tracker = EmissionsTracker(
        project_name="EHR_baseline",
        output_dir=data_dir,
        output_file="emissions_script1.csv",
        log_level="error",
        save_to_file=True,
    )
    tracker.start()
    t0 = time.perf_counter()

    for name in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Dataset: {name}")
        print(f"{'='*55}")
        X, y = load_dataset(name, data_dir)
        print(f"  Samples={len(y)}, Features={X.shape[1]}, "
              f"Class balance={y.mean():.2%}")

        dt0  = time.perf_counter()
        mccs = repeated_holdout(X, y)
        dt   = time.perf_counter() - dt0

        mean_mcc = np.mean(mccs)
        std_mcc  = np.std(mccs)
        results[name] = {"mean_MCC": mean_mcc, "std_MCC": std_mcc, "time_s": dt}

        print(f"  MCC  = {mean_mcc:.4f} ± {std_mcc:.4f}")
        print(f"  Time = {dt:.2f} s")

    elapsed   = time.perf_counter() - t0
    emissions = tracker.stop()                   # kg CO2-eq

    # Convert units
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else float("nan")
    energy_wh  = energy_kwh * 1000
    co2_g      = (emissions or 0) * 1000        # kg → g

    print(f"\n{'='*55}")
    print("  ENERGY SUMMARY (Script 1 – Baseline)")
    print(f"{'='*55}")
    print(f"  Total wall-clock time : {elapsed:.2f} s")
    print(f"  Energy consumed       : {energy_wh:.6f} Wh")
    print(f"  CO2-eq emitted        : {co2_g:.6f} g")

    # Save summary
    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(data_dir, "results_script1.csv"))
    print("\n  Results saved to results_script1.csv")

    return results, energy_wh, co2_g, elapsed


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    main(data_dir)
