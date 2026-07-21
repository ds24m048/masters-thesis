"""
This script downloads GFS weather forecast data for a specified month and day range, processes it to compute 
spatially averaged weather features, and saves the results as Parquet files. 
The spatial averaging weights are calculated using the data generated in the notebook "power_units.ipynb".
It also handles cleanup of downloaded files and logs runtime information.
"""


from pathlib import Path
import calendar
from datetime import datetime, timedelta
import time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import xarray as xr

#0710
# ── Configuration ────────────────────────────────────────────────────────────
MONTH = "202305"       # Target month to download (YYYYMM)
START_DAY = 10       # First day of the month to process (1-based, useful for resuming)
RUN = "06"             # GFS model run hour in UTC (00, 06, 12, or 18)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"  # NOAA GFS public S3 bucket
ROOT_DIR = Path(__file__).resolve().parent
POWER_DIR = ROOT_DIR.parent / "Marktstammdatenregister"  # Wind/solar plant registry data
OUTPUT_DIR = ROOT_DIR / "processed"                      # Output directory for Parquet files


def forecast_hours_for_date(date: str, run: str) -> range:
    """
    Compute the GFS forecast hour range that covers one full German calendar day.

    GFS forecasts are indexed by forecast hour (hours since the model run start).
    Since we want data for a full day in CET/CEST (00:00–23:00 local time),
    we need to account for the UTC offset to determine which forecast hours to download.

    Example: For CET (UTC+1) and run "06", midnight local = 23:00 UTC,
             so forecast hour = 23 - 6 = 17, and we download hours 17–40.

    Args:
        date: Date string in YYYYMMDD format (the day we want weather data for).
        run:  GFS model run hour as string (e.g., "06").

    Returns:
        Range of 24 forecast hours covering the full local-time day.
    """
    tz = ZoneInfo("Europe/Berlin")
    run_hour = int(run)

    # We need midnight at the START of the target day in local time.
    # Adding 1 day then taking midnight gives us the end of the target day,
    # which equals the start of the next day — this is used to compute the offset.
    target = datetime.strptime(date, "%Y%m%d") + timedelta(days=1)
    local_midnight = datetime(target.year, target.month, target.day, 0, 0, tzinfo=tz)

    # Convert local midnight to a forecast hour relative to the model run
    utc_offset = int(local_midnight.utcoffset().total_seconds() / 3600)  # +1 for CET, +2 for CEST
    f_start = 24 - utc_offset - run_hour
    return range(f_start, f_start + 24)


def month_dates(month: str) -> list[str]:
    """Return a list of all date strings (YYYYMMDD) for the given month (YYYYMM)."""
    year = int(month[:4])
    mon = int(month[4:6])
    _, n_days = calendar.monthrange(year, mon)
    return [f"{year:04d}{mon:02d}{day:02d}" for day in range(1, n_days + 1)]


def build_weights(ds: xr.Dataset, df_power: pd.DataFrame) -> xr.DataArray:
    """
    Build capacity-based spatial weights for averaging weather data across grid points.

    Each power plant is mapped to its nearest GFS grid point. The total installed
    capacity (Nettonennleistung) at each grid point is summed up and normalized
    to create weights that sum to 1. Grid points with more installed capacity
    contribute more to the spatial average.

    Args:
        ds:       GFS dataset with latitude/longitude coordinates (0.25° grid).
        df_power: Power plant DataFrame with columns Breitengrad, Laengengrad,
                  and Nettonennleistung (from Marktstammdatenregister).

    Returns:
        Normalized weight array aligned to the GFS grid (same shape as ds lat/lon).
    """
    # Snap each plant's coordinates to the nearest GFS grid point
    sel = ds.sel(
        {
            "latitude": xr.DataArray(df_power["Breitengrad"].values, dims="plant"),
            "longitude": xr.DataArray(df_power["Laengengrad"].values, dims="plant"),
        },
        method="nearest",
    )

    # Record which grid point each plant was mapped to
    df_power = df_power.copy()
    df_power["lat_gp"] = sel["latitude"].values
    df_power["lon_gp"] = sel["longitude"].values

    # Sum installed capacity per grid point
    grouped = df_power.groupby(["lat_gp", "lon_gp"])["Nettonennleistung"].sum()

    # Convert to xarray and align with the full GFS grid (fill missing grid points with 0)
    weights = (
        grouped.to_xarray()
        .rename({"lat_gp": "latitude", "lon_gp": "longitude"})
        .reindex({"latitude": ds["latitude"], "longitude": ds["longitude"]}, fill_value=0)
    )

    # Normalize so weights sum to 1
    weights = weights.fillna(0)
    s = weights.sum()
    if float(s) == 0.0:
        raise ValueError("Sum of Nettonennleistung is 0. Check mapping or filtering.")

    return weights / s


def load_and_prepare_dataset(out_path: Path) -> xr.Dataset:
    """
    Load a single GFS GRIB2 file and extract all relevant weather variables.

    The GRIB2 file contains many variables at different levels and step types.
    We use cfgrib filters to selectively read only the variables we need:
      - ssrd (sdswrf): Surface downward short-wave radiation flux [W/m²] (for solar)
      - sp:            Surface pressure [Pa]
      - tcc:           Total cloud cover [0-1]
      - u, v @100m:    Wind components at 100m hub height [m/s] → combined to wind speed
      - t2m:           Temperature at 2m above ground [K]
      - rh2m (r2):     Relative humidity at 2m [%]

    The dataset is cropped to Germany's bounding box (lat 47-56°, lon 5-16°).

    Args:
        out_path: Path to the downloaded GRIB2 file.

    Returns:
        xr.Dataset with variables: ssrd, tcc, t2m, sp, U100, rh2m,
        indexed by valid_time (UTC), latitude, longitude.
    """
    # ── Surface-level variables ──────────────────────────────────────────────
    # Solar radiation (time-averaged over the forecast step)
    ds_sfc_avg = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "avg"}, "indexpath": ""},
    )
    sdswrf = ds_sfc_avg["sdswrf"]

    # Surface pressure (instantaneous)
    ds_sfc_inst = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}, "indexpath": ""},
    )
    sp = ds_sfc_inst["sp"]

    # ── Atmosphere-level variables ───────────────────────────────────────────
    # Total cloud cover (instantaneous)
    ds_atm_inst = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "atmosphere", "stepType": "instant"}, "indexpath": ""},
    )
    tcc = ds_atm_inst["tcc"]

    # ── Height-above-ground variables ────────────────────────────────────────
    # Wind u-component at 100m (hub height for wind turbines)
    ds_hag_inst_u = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "stepType": "instant", "shortName": "u"},
            "indexpath": "",
        },
    )
    # Wind v-component at 100m
    ds_hag_inst_v = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "stepType": "instant", "shortName": "v"},
            "indexpath": "",
        },
    )
    # Temperature at 2m above ground
    ds_hag_inst_t = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "stepType": "instant", "shortName": "2t"},
            "indexpath": "",
        },
    )
    # Relative humidity at 2m above ground
    ds_hag_inst_rh = xr.open_dataset(
        out_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "stepType": "instant", "shortName": "2r"},
            "indexpath": "",
        },
    )

    # Compute wind speed at 100m from u and v components: speed = sqrt(u² + v²)
    u100 = ds_hag_inst_u["u"].sel(heightAboveGround=100)
    v100 = ds_hag_inst_v["v"].sel(heightAboveGround=100)
    u100_speed = np.hypot(u100, v100).drop_vars("heightAboveGround")

    t2m = ds_hag_inst_t["t2m"].drop_vars("heightAboveGround")
    rh2m = ds_hag_inst_rh["r2"].drop_vars("heightAboveGround")

    # ── Merge and clean up ───────────────────────────────────────────────────
    ds = xr.Dataset(data_vars={"ssrd": sdswrf, "tcc": tcc, "t2m": t2m, "sp": sp, "U100": u100_speed, "rh2m": rh2m})
    ds = ds.drop_vars(["time", "step", "surface", "atmosphere"], errors="ignore")

    # Crop to Germany's approximate bounding box
    ds = ds.sel(latitude=slice(56, 47), longitude=slice(5, 16))

    # Ensure valid_time is a proper UTC-aware 1D coordinate
    # (some GRIB files return it as scalar, expand_dims needs an iterable)
    vt = pd.to_datetime(np.atleast_1d(ds["valid_time"].values), utc=True)
    ds = ds.drop_vars("valid_time", errors="ignore")
    ds = ds.expand_dims(valid_time=vt)

    return ds


def save_outputs(date: str, df_wind: pd.DataFrame, df_solar: pd.DataFrame) -> None:
    """Save the processed wind and solar DataFrames as date-stamped Parquet files."""
    paths = {
        "wind": OUTPUT_DIR / "wind" / f"{date}.parquet",
        "solar": OUTPUT_DIR / "solar" / f"{date}.parquet",
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    df_wind.sort_index().to_parquet(paths["wind"])
    df_solar.sort_index().to_parquet(paths["solar"])

    print("Saved:")
    for key, path in paths.items():
        print(f" - {key}: {path}")


def process_day(date: str, run: str, df_wind_power: pd.DataFrame, df_solar_power: pd.DataFrame) -> None:
    """
    Download, process, and save all 24 forecast hours for a single day.

    For each forecast hour:
      1. Download the GRIB2 file from NOAA S3
      2. Extract and prepare the weather variables
      3. Compute capacity-weighted spatial averages for wind and solar

    After all hours are processed, results are concatenated and saved as Parquet.
    Downloaded GRIB2 files are cleaned up in the finally block (even on errors).

    Args:
        date:          Date string (YYYYMMDD) for which to download forecasts.
        run:           GFS model run hour (e.g., "06").
        df_wind_power: Wind plant registry DataFrame (for weight computation).
        df_solar_power: Solar plant registry DataFrame (for weight computation).
    """
    day_start = time.perf_counter()

    # Weights are computed once from the first forecast hour's grid, then reused
    weights_wind = None
    weights_solar = None

    wind_frames = []
    solar_frames = []
    downloaded_files = []

    try:
        for fh_int in forecast_hours_for_date(date, run):
            # ── Download GRIB2 file for this forecast hour ───────────────
            fh = f"{fh_int:03d}"
            file_name = f"gfs.t{run}z.pgrb2.0p25.f{fh}"
            url = f"{BASE_URL}/gfs.{date}/{run}/atmos/{file_name}"
            out_path = ROOT_DIR / file_name

            print(f"[{date}] Downloading: {url}")
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()

            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"[{date}] Saved to: {out_path}")
            downloaded_files.append(out_path)

            # ── Extract variables and compute spatial averages ───────────
            ds = load_and_prepare_dataset(out_path)

            # Build weights only once (the grid is identical for all forecast hours)
            if weights_wind is None:
                weights_wind = build_weights(ds, df_wind_power)
                weights_solar = build_weights(ds, df_solar_power)

            # Capacity-weighted spatial average → one value per variable per hour
            ds_mean_wind = ds.weighted(weights_wind).mean(dim=["latitude", "longitude"], skipna=True)
            wind_frames.append(ds_mean_wind.to_dataframe())

            ds_mean_solar = ds.weighted(weights_solar).mean(dim=["latitude", "longitude"], skipna=True)
            solar_frames.append(ds_mean_solar.to_dataframe())

        # ── Concatenate all hours and save ───────────────────────────────
        df_mean_wind = pd.concat(wind_frames)
        df_mean_solar = pd.concat(solar_frames)

        save_outputs(date, df_mean_wind, df_mean_solar)
    finally:
        # ── Cleanup: delete downloaded GRIB2 files and their index files ─
        to_delete = []
        for path in downloaded_files:
            to_delete.append(path)
            to_delete.extend(path.parent.glob(path.name + "*.idx"))

        deleted = []
        for path in to_delete:
            if path.exists():
                path.unlink()
                deleted.append(path)

        if deleted:
            print(f"[{date}] Deleted downloaded files:")
            for path in deleted:
                print(f" - {path}")
        else:
            print(f"[{date}] No matching downloaded files found to delete.")

        elapsed_s = time.perf_counter() - day_start
        elapsed_h = int(elapsed_s // 3600)
        elapsed_m = int((elapsed_s % 3600) // 60)
        elapsed_sec = elapsed_s % 60
        print(f"[{date}] Total runtime: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_sec:05.2f}")


def main() -> None:
    """
    Main entry point: process all days in the configured month.

    Reads the configuration from the module-level constants (MONTH, START_DAY, RUN),
    loads the plant registry data, and iterates over each day in the month.
    """
    month_start = time.perf_counter()

    month = MONTH
    start_day = START_DAY
    run = RUN

    # Load plant registry data (coordinates + installed capacity per plant)
    df_wind_power = pd.read_parquet(POWER_DIR / "power_wind.parquet")
    df_solar_power = pd.read_parquet(POWER_DIR / "power_solar.parquet")

    # Build list of dates and optionally skip already-processed days via START_DAY
    dates = month_dates(month)
    if not 1 <= start_day <= len(dates):
        raise ValueError(f"START_DAY must be between 1 and {len(dates)} for month {month}.")
    dates = dates[start_day - 1 :]
    print(f"Processing month {month} from day {start_day} ({len(dates)} days)")

    for date in dates:
        print(f"\n=== Start day {date} ===")
        process_day(date, run, df_wind_power, df_solar_power)

    elapsed_s = time.perf_counter() - month_start
    elapsed_h = int(elapsed_s // 3600)
    elapsed_m = int((elapsed_s % 3600) // 60)
    elapsed_sec = elapsed_s % 60
    print(f"\nTotal runtime for month {month}: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_sec:05.2f}")


if __name__ == "__main__":
    main()
