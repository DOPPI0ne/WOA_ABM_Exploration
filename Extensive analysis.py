import numpy as np
import pandas as pd
from pathlib import Path
import time
import multiprocessing as mp
from tqdm import tqdm

from ABM import IndividualModel

# ── Configurazione ─────────────────────────────────────────────────────────────
N_SIMS      = 50000
T           = 2500       # safety cap
N_WORKERS   = 6
CHUNK_SIZE  = 500
OUTPUT_DIR  = Path(r"D:\WOA_Exploration\doc")
RANDOM_SEED = 42

# ── Range parametrici ─────────────────────────────────────────────────────────
PARAM_RANGES = {
    "n":   (0.0001,  0.040),
    "m":   (0.0001,  0.015),
    "P_0": (5,       200),
    "c":   (1.0e-04, 8.0e-04),
    "d":   (1.0e-08, 6.0e-08),
    "k":   (10.0,    60.0),
    "R":   (1000.0,  8000.0),
    "z":   (2,       30),
}

def build_values(vmin, vmax, n=20, as_int=False):
    vals = np.linspace(vmin, vmax, n)
    if as_int:
        vals = np.unique(np.round(vals).astype(int))
    return vals.tolist()

VALUE_SPACE = {
    "n":   build_values(*PARAM_RANGES["n"]),
    "m":   build_values(*PARAM_RANGES["m"]),
    "P_0": build_values(*PARAM_RANGES["P_0"], as_int=True),
    "c":   build_values(*PARAM_RANGES["c"]),
    "d":   build_values(*PARAM_RANGES["d"]),
    "k":   build_values(*PARAM_RANGES["k"]),
    "R":   build_values(*PARAM_RANGES["R"]),
    "z":   build_values(*PARAM_RANGES["z"], as_int=True),
}

# ── Slope al punto di flesso ──────────────────────────────────────────────────
def compute_inflection_slope(arr, t_peak):
    """
    Trova il punto di massimo tasso di collasso post-picco.
    Derivata prima: dP[t] = P[t+1] - P[t]
    Slope normalizzata: dP[t_flesso] / P[t_flesso]
    """
    post = arr[t_peak:]
    if len(post) < 3:
        return float("nan"), t_peak, float("nan")

    dP = np.diff(post)
    t_flesso_rel = int(np.argmin(dP))
    t_flesso_abs = t_peak + t_flesso_rel
    dP_min       = float(dP[t_flesso_rel])
    P_at_flesso  = float(post[t_flesso_rel])

    slope_norm = dP_min / P_at_flesso if P_at_flesso > 0 else float("nan")

    return slope_norm, t_flesso_abs, dP_min

# ── Summary ───────────────────────────────────────────────────────────────────
def compute_summary(sim_id, params, pop_ts):
    arr = np.asarray(pop_ts, dtype=float)

    P_peak  = float(np.max(arr))
    t_peak  = int(np.argmax(arr))
    P_final = float(arr[-1])
    t_final = len(arr) - 1

    full_collapse = int(P_final == 0)
    t_extinction  = int(np.argmax(arr == 0)) if full_collapse else None

    post_peak = arr[t_peak:]
    if len(post_peak) > 1:
        t_min_rel   = int(np.argmin(post_peak))
        P_min_after = float(post_peak[t_min_rel])
        t_min_after = t_peak + t_min_rel
    else:
        P_min_after = P_final
        t_min_after = t_final

    if P_min_after > 0:
        recovered = int(P_final > P_min_after * 1.10)
    else:
        recovered = int(P_final > 5)

    collapse_metric = float(np.clip(P_final / P_peak, 0.0, 1.0)) if P_peak > 0 else 0.0

    slope_inflection, t_flesso, dP_min = compute_inflection_slope(arr, t_peak)

    row = {"sim_id": sim_id}
    row.update(params)
    row.update({
        "P_peak":           P_peak,
        "t_peak":           t_peak,
        "P_min_after":      P_min_after,
        "t_min_after":      t_min_after,
        "P_final":          P_final,
        "t_final":          t_final,
        "t_extinction":     t_extinction,
        "full_collapse":    full_collapse,
        "recovered":        recovered,
        "collapse_metric":  collapse_metric,
        "collapse_flag":    int(collapse_metric < 0.50),
        "slope_inflection": slope_inflection,
        "t_flesso":         t_flesso,
        "dP_min":           dP_min,
    })
    return row

# ── Simulazione singola ───────────────────────────────────────────────────────
def run_single(args):
    sim_id, params, seed = args

    model = IndividualModel(
        T,
        float(params["c"]), float(params["m"]), float(params["n"]),
        float(params["d"]), float(params["k"]), float(params["R"]),
        int(params["P_0"]), 0.0, int(params["z"]),
        engine="numba"
    )
    model.setup()
    model._seed = seed
    model.run()

    pop_ts = np.asarray(model.P_ts, dtype=float)

    zero_idx = np.argmax(pop_ts == 0)
    if zero_idx > 0 and pop_ts[zero_idx] == 0:
        pop_ts = pop_ts[:zero_idx + 1]

    return compute_summary(sim_id, params, pop_ts)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"abm_tipping_summary_{N_SIMS}.csv"

    rng = np.random.default_rng(RANDOM_SEED)
    all_params = [
        {p: rng.choice(VALUE_SPACE[p]) for p in VALUE_SPACE}
        for _ in range(N_SIMS)
    ]
    all_seeds = rng.integers(0, 2_147_483_647, size=N_SIMS).tolist()

    args_list = [
        (sim_id, all_params[sim_id], int(all_seeds[sim_id]))
        for sim_id in range(N_SIMS)
    ]

    print(f"N_SIMS={N_SIMS} | T={T} | N_WORKERS={N_WORKERS}")
    t0 = time.time()

    summary_rows = []

    with mp.Pool(processes=N_WORKERS) as pool:
        for i, row in enumerate(tqdm(
            pool.imap(run_single, args_list, chunksize=10),
            total=N_SIMS,
            desc="Simulazioni",
            unit="sim"
        )):
            summary_rows.append(row)
            if (i + 1) % CHUNK_SIZE == 0:
                pd.DataFrame(summary_rows).to_csv(out_path, index=False)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved: {out_path}  ({len(df)} rows)")
    print(f"Tempo totale: {elapsed / 60:.1f} minuti")

    print("\n=== Quick summary ===")
    print(f"Full collapse (P→0):      {df.full_collapse.mean()*100:.1f}%")
    print(f"Recovered after collapse: {df.recovered.mean()*100:.1f}%")
    print(f"Collapse flag (α=0.50):   {df.collapse_flag.mean()*100:.1f}%")
    print(f"slope_inflection (mean):  {df.slope_inflection.mean():.4f}")
    print(f"slope_inflection (median):{df.slope_inflection.median():.4f}")

    print("\n=== Collapse rate per bin di n ===")
    df["n_bin"] = pd.cut(df["n"], bins=10)
    print(df.groupby("n_bin", observed=True)[
        ["full_collapse", "recovered", "collapse_flag"]
    ].mean().round(3).to_string())