import argparse
import json
from pathlib import Path
from typing import Dict, Set

import geopandas as gpd
import earthaccess
from shapely.geometry import box
from shapely.strtree import STRtree

import yaml
from shapely.geometry import box, Polygon
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
    

    canals_path = Path(config["base_grain_dir"]) / f"{region}_GRAIN_v.1.0.parquet"

    buffer_m = config["buffer_m"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    short_name = config["pixc_short_name"]

    planning_dir = run_root / config["planning_dir"]
    download_dir = run_root / config["download_dir"]
    planning_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    canals = gpd.read_parquet(canals_path)
    start = args.canal_start
    end = args.canal_end
    

    if start is not None and end is not None:
        canals = canals.sort_values("grain_id").iloc[start:end]
        print(f"[INFO] Processing canals {start}:{end}")

    # print("Number of canals to process:", len(canals))
    if canals.crs is None:
        raise RuntimeError("Canals CRS missing")
    
    canals = canals.to_crs(4326)

    min_lon, min_lat, max_lon, max_lat = canals.total_bounds
    region_bbox = (float(min_lon), float(min_lat),
                float(max_lon), float(max_lat))

    earthaccess.login()

    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=region_bbox,
        temporal=(start_date, end_date),
        cloud_hosted=False,
    )

    if not results:
        print("No PIXC found.")
        return

    print(f"[INFO] Total granules returned: {len(results)}")

    # Build canal spatial index
    canals_proj = canals.to_crs(3857)
    canal_geoms = list(canals_proj.geometry)
    canal_ids = list(canals_proj["grain_id"])
    tree = STRtree(canal_geoms)

    pixc_to_grains: Dict[str, list] = {}
    unique_results = {}

    for r in results:
        granule_name = r["meta"]["native-id"]

        # Extract granule geometry
        geom_info = (
            r.get("umm", {})
            .get("SpatialExtent", {})
            .get("HorizontalSpatialDomain", {})
            .get("Geometry", {})
        )

        granule_geom = None

        if "BoundingRectangles" in geom_info:
            bbox = geom_info["BoundingRectangles"][0]
            granule_geom = box(
                bbox["WestBoundingCoordinate"],
                bbox["SouthBoundingCoordinate"],
                bbox["EastBoundingCoordinate"],
                bbox["NorthBoundingCoordinate"],
            )

        elif "GPolygons" in geom_info:
            coords = geom_info["GPolygons"][0]["Boundary"]["Points"]
            lonlat = [(p["Longitude"], p["Latitude"]) for p in coords]
            granule_geom = Polygon(lonlat)

        if granule_geom is None:
            continue

        granule_geom_proj = (
            gpd.GeoSeries([granule_geom], crs=4326)
            .to_crs(3857)
            .iloc[0]
        )

        candidate_idxs = tree.query(granule_geom_proj)

        intersecting_grains = []
        for idx in candidate_idxs:
            if canal_geoms[idx].intersects(granule_geom_proj):
                intersecting_grains.append(str(canal_ids[idx]))

        
        if intersecting_grains:
            pixc_to_grains[granule_name] = intersecting_grains
            unique_results[granule_name] = r
    # Save mapping
    out_json = planning_dir / "pixc_to_grains.json"
    with open(out_json, "w") as f:
        json.dump(pixc_to_grains, f, indent=2)

    # Save unique list
    out_unique = planning_dir / "unique_pixc_granules.txt"
    out_unique.write_text("\n".join(sorted(unique_results.keys())) + "\n")

    print("[INFO] Preparing download list...\n Total granules:", len(unique_results))

    # Only download missing files
    to_download = []

    for granule_name, r in unique_results.items():
        expected_file = download_dir / f"{granule_name}.nc"
        if not expected_file.exists():
            to_download.append(r)

    print(f"[INFO] Granules missing locally: {len(to_download)}")

    if to_download:
        earthaccess.download(to_download, str(download_dir))
    else:
        print("[INFO] All granules already present. Skipping download.")

        print("[OK] Planning and download complete.")


if __name__ == "__main__":
    main()