import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
import yaml


def robust_sigma(residuals):
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 1.4826 * mad


def compute_metrics(df, config):
    BIN_WIDTH = config["bin_width_m"]
    MIN_BINS = config["min_bins_required"]
    CLASS_ALLOWED = set(config["classification"])
    CT_MIN = config["cross_track_min_m"]
    CT_MAX = config["cross_track_max_m"]

    mask = (
        df["classification"].isin(CLASS_ALLOWED) &
        (df["cross_track"].abs() >= CT_MIN) &
        (df["cross_track"].abs() <= CT_MAX)
    )

    df = df.loc[mask].copy()
    if df.empty:
        return None

    df["wse"] = df["height"] - df["geoid"]
    df = df[np.isfinite(df["wse"])]
    if df.empty:
        return None

    df["bin"] = (df["d_m"] // BIN_WIDTH).astype(int)
    grouped = df.groupby("bin")

    bin_centers = []
    medians = []

    for b, g in grouped:
        bin_centers.append((b + 0.5) * BIN_WIDTH)
        medians.append(np.median(g["wse"]))

    if len(bin_centers) < MIN_BINS:
        return None

    d_vals = np.array(bin_centers)
    wse_vals = np.array(medians)

    slope, intercept, _, _ = theilslopes(wse_vals, d_vals)

    residuals = wse_vals - (intercept + slope * d_vals)
    sigma = robust_sigma(residuals)

    canal_length = d_vals.max() - d_vals.min()

     
    # Coverage + Continuity
     
    bins_sorted = np.sort(df["bin"].unique())

    min_bin = bins_sorted.min()
    max_bin = bins_sorted.max()
    total_possible_bins = max_bin - min_bin + 1

    coverage_frac = len(bins_sorted) / total_possible_bins

    # Longest contiguous run
    max_run = 1
    current_run = 1

    for i in range(1, len(bins_sorted)):
        if bins_sorted[i] == bins_sorted[i - 1] + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    contig_frac = max_run / total_possible_bins

     
    # Signal-to-noise
     
    abs_slope = abs(slope)
    signal_amp = abs_slope * canal_length

    if sigma > 0:
        slope_snr = signal_amp / sigma
    else:
        slope_snr = np.nan

    return {
        "n_bins": len(d_vals),
        "slope_m_per_m": slope,
        "abs_slope_m_per_m": abs_slope,
        "residual_sigma_robust_m": sigma,
        "canal_length_m": canal_length,
        "coverage_frac": coverage_frac,
        "contig_frac": contig_frac,
        "slope_snr": slope_snr,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--canal_start", required=True, type=int)
    parser.add_argument("--canal_end", required=True, type=int)
    return parser.parse_args()

def main():
    args = parse_args()

     
    # LOAD CONFIG
     
    config = yaml.safe_load(open(args.config))

    run_base = Path(config["run_base_dir"])
    run_root = run_base / f"{args.region}_from_{args.canal_start}_to_{args.canal_end}"

    input_dir = run_root / config["extracted_points_dir"]
    out_dir = run_root / config["metrics_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

     
    # FIND ALL PARQUET FILES (RECURSIVELY)
     
    parquet_files = sorted(input_dir.rglob("*.parquet"))

    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found under {input_dir}. "
            "Extraction step may have failed."
        )

    print(f"[INFO] Found {len(parquet_files)} parquet files.")

     
    # COMPUTE METRICS PER GRAIN
     
    records = []

    for parquet_file in parquet_files:
        grain_id = parquet_file.stem

        try:
            df = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"[WARNING] Failed to read {parquet_file}: {e}")
            continue

        result = compute_metrics(df, config)

        if result is not None:
            result["grain_id"] = grain_id
            records.append(result)

    if not records:
        raise RuntimeError(
            "No valid metric records were computed. "
            "Check filtering thresholds or extraction output."
        )

     
    # WRITE OUTPUT
     
    out_csv = out_dir / "wse_metrics.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)

    print(f"[OK] Metrics computed for {len(records)} grains.")
    print(f"[OK] Output written to {out_csv}")

if __name__ == "__main__":
    main()