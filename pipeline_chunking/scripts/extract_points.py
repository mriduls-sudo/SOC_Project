#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from shapely import points, distance, line_locate_point
from shapely.ops import unary_union
from pyproj import CRS, Transformer


# --------------------------------------------------------
# ARGUMENTS
# --------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--region", required=True)

    # GLOBAL RUN RANGE
    p.add_argument("--run_start", type=int, required=True)
    p.add_argument("--run_end", type=int, required=True)

    # CHUNK RANGE
    p.add_argument("--canal_start", type=int, required=True)
    p.add_argument("--canal_end", type=int, required=True)

    p.add_argument("--chunk_id", required=True)
    return p.parse_args()

def get_utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    if lat >= 0:
        return CRS.from_epsg(32600 + zone)
    return CRS.from_epsg(32700 + zone)


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))

    region = args.region
    chunk_id = args.chunk_id

    run_base = Path(config["run_base_dir"])
    run_root = run_base / f"{region}_from_{args.run_start}_to_{args.run_end}"

    canals_path = Path(config["base_grain_dir"]) / f"{region}_GRAIN_v.1.0.parquet"
    planning_json = run_root / config["planning_dir"] / "pixc_to_grains.json"
    download_dir = run_root / config["download_dir"]

    # ⬇ chunk-specific output directory
    out_dir = (
        run_root
        / config["extracted_points_dir"]
        / f"chunk_{chunk_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer_m = float(config["buffer_m"])

    # --------------------------------------------------------
    # LOAD CANALS (ONLY THIS CHUNK)
    # --------------------------------------------------------

    canals = gpd.read_parquet(canals_path)

    if canals.crs is None:
        raise RuntimeError("Canals file has no CRS")

    if canals.crs.to_epsg() != 4326:
        canals = canals.to_crs(4326)

    canals = canals.sort_values("grain_id")

    canals = canals.iloc[args.canal_start:args.canal_end]
    print(f"[INFO] Chunk {chunk_id}: canals {args.canal_start}:{args.canal_end}")

    centroid = canals.geometry.union_all().centroid
    target_crs = get_utm_crs_from_lonlat(centroid.x, centroid.y)
    canals = canals.to_crs(target_crs)

    print(f"[INFO] Using CRS: {target_crs}")

    canal_lookup = {
        str(row["grain_id"]): row.geometry
        for _, row in canals.iterrows()
    }
    canal_ids_set = set(canal_lookup.keys())

    # --------------------------------------------------------
    # LOAD PLANNING JSON
    # --------------------------------------------------------

    with open(planning_json) as f:
        pixc_to_grains = json.load(f)

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    writers = {}

    # --------------------------------------------------------
    # GRANULE LOOP
    # --------------------------------------------------------

    for pixc_path in sorted(download_dir.glob("*.nc")):
        granule_name = pixc_path.stem

        if granule_name not in pixc_to_grains:
            continue

        relevant_grains = [
            str(g) for g in pixc_to_grains[granule_name]
            if str(g) in canal_ids_set
        ]

        if not relevant_grains:
            continue

        print(f"[INFO] {chunk_id} → {granule_name}")

        try:
            ds = xr.open_dataset(pixc_path, group="pixel_cloud", decode_times=False)
        except OSError:
            print(f"Corrupted PIXC file skipped: {pixc_path}")
            continue

        lat = ds["latitude"].values.ravel()
        lon = ds["longitude"].values.ravel()
        height = ds["height"].values.ravel()
        geoid = ds["geoid"].values.ravel()
        classification = ds["classification"].values.ravel()

        water_frac = ds["water_frac"].values.ravel()
        water_frac_uncert = ds["water_frac_uncert"].values.ravel()
        cross_track = ds["cross_track"].values.ravel()

        interferogram_qual = ds["interferogram_qual"].values.ravel()
        classification_qual = ds["classification_qual"].values.ravel()
        geolocation_qual = ds["geolocation_qual"].values.ravel()
        sig0_qual = ds["sig0_qual"].values.ravel()

        ds.close()

        valid = np.isfinite(lat) & np.isfinite(lon)
        if not valid.any():
            continue

        lat = lat[valid]
        lon = lon[valid]

        height = height[valid]
        geoid = geoid[valid]
        classification = classification[valid]

        water_frac = water_frac[valid]
        water_frac_uncert = water_frac_uncert[valid]
        cross_track = cross_track[valid]

        interferogram_qual = interferogram_qual[valid]
        classification_qual = classification_qual[valid]
        geolocation_qual = geolocation_qual[valid]
        sig0_qual = sig0_qual[valid]

        xs, ys = transformer.transform(lon, lat)

        combined_geom = unary_union([canal_lookup[g] for g in relevant_grains])
        minx, miny, maxx, maxy = combined_geom.buffer(buffer_m).bounds

        mask_bbox = (
            (xs >= minx) & (xs <= maxx) &
            (ys >= miny) & (ys <= maxy)
        )

        if not mask_bbox.any():
            continue

        xs_sub = xs[mask_bbox]
        ys_sub = ys[mask_bbox]
        pts = points(xs_sub, ys_sub)

        for grain_id in relevant_grains:
            canal_geom = canal_lookup[grain_id]

            dist_all = distance(pts, canal_geom)
            mask_buffer = dist_all <= buffer_m
            if not mask_buffer.any():
                continue

            pts_keep = points(xs_sub[mask_buffer], ys_sub[mask_buffer])
            d_vals = line_locate_point(canal_geom, pts_keep)

            df_out = pd.DataFrame({
                "grain_id": grain_id,
                "pixc_granule_id": granule_name,
                "x": xs_sub[mask_buffer],
                "y": ys_sub[mask_buffer],
                "d_m": d_vals,
                "dist_to_canal_m": dist_all[mask_buffer],
                "height": height[mask_bbox][mask_buffer],
                "geoid": geoid[mask_bbox][mask_buffer],
                "classification": classification[mask_bbox][mask_buffer],
                "water_frac": water_frac[mask_bbox][mask_buffer],
                "water_frac_uncert": water_frac_uncert[mask_bbox][mask_buffer],
                "cross_track": cross_track[mask_bbox][mask_buffer],
                "interferogram_qual": interferogram_qual[mask_bbox][mask_buffer],
                "classification_qual": classification_qual[mask_bbox][mask_buffer],
                "geolocation_qual": geolocation_qual[mask_bbox][mask_buffer],
                "sig0_qual": sig0_qual[mask_bbox][mask_buffer],
            })

            table = pa.Table.from_pandas(df_out, preserve_index=False)
            out_path = out_dir / f"{grain_id}.parquet"

            if grain_id not in writers:
                writers[grain_id] = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True
                )

            writers[grain_id].write_table(table)

            del df_out

        gc.collect()

    for writer in writers.values():
        writer.close()

    flag_path = out_dir / "extraction_complete.flag"
    flag_path.write_text("done\n")

    print(f"[INFO] Chunk {chunk_id} completed.")


if __name__ == "__main__":
    main()