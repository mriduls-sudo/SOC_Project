import argparse
from datetime import datetime, UTC
import time
from pathlib import Path
import numpy as np
from scipy.stats import theilslopes
import pandas as pd
import ee
from google.cloud import storage
import yaml


# Helpers


def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def wait_for_task(task):
    print("[INFO] Waiting for GEE export...")
    while task.active():
        print("   State:", task.status()["state"])
        time.sleep(30)

    status = task.status()
    if status["state"] != "COMPLETED":
        raise RuntimeError(f"GEE task failed: {status}")

    print("[OK] Export completed")


def download_from_gcs(bucket, blob_name, out_path, project):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = storage.Client(project=project)
    bucket = client.bucket(bucket)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise RuntimeError(f"GCS object not found: gs://{bucket.name}/{blob_name}")

    blob.download_to_filename(out_path)
    print(f"[OK] Downloaded {out_path}")


# DEM Sampling and Slope Comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--canal_start", required=True, type=int)
    parser.add_argument("--canal_end", required=True, type=int)
    
    return parser.parse_args()

def main():
    args = parse_args()

    config = yaml.safe_load(open(args.config))
    region = args.region
    run_base = Path(config["run_base_dir"])
    run_root = run_base / f"{region}_from_{args.canal_start}_to_{args.canal_end}"
    gee_asset = f"{config['base_gee_asset']}/{region}_GRAIN_canals"
    

    metrics_csv = run_root / config["metrics_dir"] / "wse_metrics.csv"
    if not metrics_csv.exists():
        raise RuntimeError("Missing wse_metrics.csv")

    df = pd.read_csv(metrics_csv)
    canal_ids = df["grain_id"].dropna().unique().tolist()

    if not canal_ids:
        print("[SKIP] No canals for DEM sampling")
        return

    print(f"[INFO] Sampling DEM along {len(canal_ids)} canals")

    init_ee()

    
    # Load canal asset from GEE and filter to target canals
    
    canals = ee.FeatureCollection(gee_asset)

    target_canals = canals.filter(
        ee.Filter.inList("grain_id", canal_ids)
    )

    print("[INFO] GEE canal count:",
          target_canals.size().getInfo())

    
    # Build points along canal
    
    SPACING_M = config["dem_spacing_m"]
    DEM_SCALE = config["dem_scale_m"]

    def points_along(feature):
        geom = ee.Geometry(feature.geometry())
        length = geom.length()

        dists = ee.List.sequence(0, length, SPACING_M)
        cut = geom.cutLines(dists)
        segs = ee.List(cut.coordinates())

        pts = segs.map(
            lambda seg: ee.Feature(
                ee.Geometry.Point(ee.List(seg).get(0)),
                {"grain_id": feature.get("grain_id")}
            )
        )

        pts = ee.List(pts).zip(dists).map(
            lambda z: ee.Feature(ee.List(z).get(0))
            .set("dist_m", ee.List(z).get(1))
        )

        return ee.FeatureCollection(pts)

    sample_points = target_canals.map(points_along).flatten()

    
    # DEM collection choice and sampling
    
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .mosaic()
        .select("DEM")
    )

    dem_samples = dem.sampleRegions(
        collection=sample_points,
        scale=DEM_SCALE,
        geometries=False,
    )

    
    # Export
    
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    export_name = f"{region}_dem_samples_{timestamp}"

    task = ee.batch.Export.table.toCloudStorage(
        collection=dem_samples,
        description=export_name,
        bucket=config["gcs_bucket"],
        fileNamePrefix=export_name,
        fileFormat="CSV",
    )

    task.start()
    print("[INFO] GEE export started")

    wait_for_task(task)

    
    # Download locally
    
    local_out = run_root / config["metrics_dir"] / "dem_samples.csv"

    download_from_gcs(
        bucket=config["gcs_bucket"],
        blob_name=f"{export_name}.csv",
        out_path=local_out,
        project=config["gcp_project"],
    )

    print("[OK] DEM sampling complete.")

        
    # Compute DEM slope per canal
    
    

    TOL = float(config.get("slope_sign_tolerance", 1e-6))

    def slope_sign(x, tol=TOL):
        if pd.isna(x):
            return np.nan

        x = float(x)
        if abs(x) < tol:
            return 0
        return np.sign(x)

    dem_df = pd.read_csv(local_out)

    if dem_df.empty:
        print("[WARN] DEM sample file is empty")
        return

    slope_results = []

    for grain_id, g in dem_df.groupby("grain_id"):

        g = g.sort_values("dist_m")

        d_vals = g["dist_m"].values
        elev_vals = g["DEM"].values

        # Require minimum number of DEM points
        if len(d_vals) < 8:
            continue

        slope, intercept, _, _ = theilslopes(elev_vals, d_vals)

        slope_results.append({
            "grain_id": grain_id,
            "dem_slope_m_per_m": slope,
            "dem_slope_sign": slope_sign(slope),
        })

    dem_slope_df = pd.DataFrame(slope_results)

    
    # Merge with WSE metrics
    
    wse_df = pd.read_csv(metrics_csv)
    # enforce numeric type for slope column (in case of any parsing issues)
    wse_df["slope_m_per_m"] = pd.to_numeric(
        wse_df["slope_m_per_m"],
        errors="coerce"
    )

    merged = wse_df.merge(
        dem_slope_df,
        on="grain_id",
        how="left"
    )

    merged["wse_slope_sign"] = merged["slope_m_per_m"].apply(
        lambda x: slope_sign(x)
    )

    merged["slope_sign_match"] = (
        merged["wse_slope_sign"] ==
        merged["dem_slope_sign"]
    )

    # For interpretability, categorize slope sign agreement
    merged["slope_sign_category"] = np.where(
        merged["slope_sign_match"],
        "match",
        "mismatch"
    )

    out_merged = run_root / config["metrics_dir"] / "wse_dem_slope_comparison.csv"
    merged.to_csv(out_merged, index=False)

    print("[OK] DEM slope comparison complete.")


if __name__ == "__main__":
    main()