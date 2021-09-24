"""Download daily CHIRPS rain data for the African continent.
Process them based on the contours of the countries provided.
Results are aggregated by Epi week (CDC week).
"""

import os
import gzip
from datetime import date, timedelta
import logging
import tempfile

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import requests
from fsspec import AbstractFileSystem
from geopandas import GeoDataFrame
from rasterstats import zonal_stats

import storage


CHIRPS_VERSION = "2.0"
CHIRPS_TIMELY = "daily"
CHIRPS_ZONE = "africa"
# Year folder added on the fly by the download_chirps_daily function.
CHIRPS_URL = f"https://data.chc.ucsb.edu/products/CHIRPS-{CHIRPS_VERSION}/{CHIRPS_ZONE}_{CHIRPS_TIMELY}/tifs/p05/"
CHIRPS_BASENAME = "chirps-v2.0.{chirps_year}.{chirps_month:0>2d}.{chirps_day:0>2d}.tif"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Download and process CHIRPS data."""
    pass


@cli.command()
def test():
    """Run test suite."""
    return pytest.main(["-x", "tests"])


@cli.command()
@click.option("--output-dir", type=str, help="output directory")
@click.option("--start", type=int, help="start year")
@click.option("--end", type=int, help="end year")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def download(output_dir: str, start: int, end: int, overwrite: bool):
    """Download raw precipitation data."""
    logger.info(f"Downloading CHIRPS data from {start} to {end} into {output_dir}.")
    download_chirps_daily(
        output_dir=output_dir, year_start=start, year_end=end, overwrite=overwrite
    )


@cli.command()
@click.option("--start", type=int, help="start year")
@click.option("--end", type=int, help="end year")
@click.option("--contours", type=str, help="path to contours file")
@click.option("--input-dir", type=str, help="raw CHIRPS data directory")
@click.option("--output-file", type=str, help="output directory")
def extract(start, end, contours, input_dir, output_file):
    """Compute zonal statistics."""

    logger.info(f"Computing zonal statistics for {contours}")

    extract_chirps_data(
        contours_file=gpd.read_file(contours),
        input_dir=input_dir,
        output_file=output_file,
        start_year=start,
        end_year=end,
    )


def provide_epi_week(year, month, day, system="CDC"):
    """Provide epidemiological week from a date using either the CDC or WHO system.

    How do we define the first week of the year/January ?

        - Epi Week begins on a Sunday and end on a Saturday.
        - The first EpiWeek ends on the first Saturday of January as long as
          the week is at least 4 days long.
        - If less than 4 days long, the EpiWeek began on the first Sunday of
          January and the days before than belong to the last week of the
          year before.
        - If at least 4 days long, the last days of December will be part of
          the first week of the next year.

    In other words, the first epidemiological week always begins on a date
    between December 29 and January 4 and ends on a date between
    January 4 and 10.

        - Number of weeks by year : 52 or 53
        - If January 1 occurs on Sunday, Monday, Tuesday or Wednesday,
          calendar week that includes January 1 will be the week 1 of the
          current year
        - If January 1 occurs on Thursday, Friday or Saturday, the calendar
          week that includes January 1 will be th last week of previous year.

    Note
    ----
    WHO system not really tested (no reference calendar found).

    Parameters
    ----------
    year : int
        Year.
    month : int
        Month.
    day : int
        Day.
    system : str, optional
        Epidemiological week system, "CDC" or "WHO".

    Returns
    -------
    epi_week : int
        Epi week number.
    epi_year : int
        Year.
    week_start_day : datetime
        Start day of the epi week.
    week_end_day : datetime
        End day of the epi week.

    Raises
    ------
    ValueError
        If epi week system is neither CDC or WHO.
    """
    systems = {"CDC": 0, "WHO": 1}
    if system.upper() not in systems.keys():
        raise ValueError("System not in {}".format(list(systems.keys())))

    x = date(year, month, day)
    week_day = (x - timedelta(days=systems[system.upper()])).isoweekday()
    week_day = 0 if week_day == 7 else week_day  # Week : Sun = 0 ; Sat = 6

    # Start the weekday on Sunday (CDC)
    week_start_day = x - timedelta(days=week_day)
    # End the weekday on Saturday (CDC)
    week_end_day = week_start_day + timedelta(days=6)

    week_start_year_day = week_start_day.timetuple().tm_yday
    week_end_year_day = week_end_day.timetuple().tm_yday

    if week_end_year_day in range(4, 11):
        epi_week = 1
    else:
        epi_week = ((week_start_year_day + 2) // 7) + 1

    if week_end_year_day in range(4, 11):
        epi_year = week_end_day.year
    else:
        epi_year = week_start_day.year

    return (epi_week, epi_year, week_start_day, week_end_day)


def provide_time_range(year_start, year_end, future=False):
    """Get pandas date range from start and end year.

    Parameters
    ----------
    year_start : int
        Start year.
    year_end : int
        End year.
    future : bool, optional
        TODO.

    Returns
    -------
    DatetimeIndex
        Pandas fixed frequency time range.
    """
    start_period = date(year_start, 1, 1)
    _, _, start_epi_day, _ = provide_epi_week(
        date.today().year, date.today().month, date.today().day
    )
    if future:
        end_period = date(year_end, 12, 31)
    else:
        end_period = (
            start_epi_day - timedelta(days=1)
            if date(year_end, 12, 31) >= start_epi_day
            else date(year_end, 12, 31)
        )  # No Future
    return pd.date_range(start_period, end_period)


def download_chirps_data(
    *, download_url: str, fs: AbstractFileSystem, output_path: str
):
    """Download .tif or .gzip file from URL."""

    logger.info(f"Downloading {download_url}")

    # If a .tif file is requested but not available, the request will be
    # successfull (status code = 200) and the server will include the location
    # of the corresponding .tif.gz in the Content-Location header.
    src_fname = download_url.split("/")[-1]
    r = requests.head(download_url)
    download_url = download_url.replace(
        src_fname, r.headers.get("content-location", src_fname)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Write the GeoTIFF in a temporary file
        tmp_file = os.path.join(tmp_dir, "raster.tif")
        r = requests.get(download_url, stream=True, timeout=2)
        r.raise_for_status()
        with open(tmp_file, "wb") as f:
            if download_url.endswith(".gz"):
                f.write(gzip.decompress(r.content))
            else:
                f.write(r.content)

        # Compress, tile and assign nodata value with Rasterio before writing
        with rasterio.open(tmp_file) as src:
            dst_profile = src.profile.copy()
            dst_profile.update(
                compress="deflate",
                predictor=3,
                zlevel=6,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                nodata=-9999,
            )
            tmp_file_2 = os.path.join(tmp_dir, "raster2.tif")
            with rasterio.open(tmp_file_2, "w", **dst_profile) as dst:
                dst.write(src.read(1), 1)

        fs.put(tmp_file_2, output_path)


def download_chirps_daily(
    *,
    output_dir: str,
    year_start: int,
    year_end: int,
    overwrite: bool = False,
):
    """Download CHIRPS daily products for a given time range."""

    output_dir = _no_ending_slash(output_dir)
    fs = storage.filesystem(output_dir)
    time_range = provide_time_range(year_start, year_end)
    time_serie = time_range.to_series().apply(
        lambda r: provide_epi_week(r.year, r.month, r.day)
    )

    for day, (epi_week, year, _, _) in time_serie.items():
        output_file = CHIRPS_BASENAME.format(
            chirps_year=day.year, chirps_month=day.month, chirps_day=day.day
        )
        output_path = f"{output_dir}/{year}-{epi_week:0>2d}/{output_file}"
        url = f"{CHIRPS_URL}{day.year}/{output_file}"

        fs.makedirs(fs.dirname(output_path))

        if fs.exists(output_path) and fs.size(output_path) > 0 and not overwrite:
            logger.info(f"Skipping {output_path}")
            continue

        download_chirps_data(download_url=url, fs=fs, output_path=output_path)


def extract_sum_epi_week(files):
    """Get weekly precipitation sum from daily precipitation rasters.

    Parameters
    ----------
    files : list of str
        Daily rasters as a list of paths.

    Return
    ------
    weekly : ndarray
        Weekly precipitation sum as a 2D numpy array.
    affine : Affine
        Raster affine transformation.
    """
    with rasterio.open(files[0]) as src:
        affine = src.transform

    daily_rasters = []
    for f in files:
        with rasterio.open(f) as src:
            daily_rasters.append(src.read(1))

    weekly = np.nansum(daily_rasters, axis=0)
    return weekly, affine


def extract_chirps_data(
    *,
    contours_file: str,
    input_dir: str,
    output_file: str,
    start_year: int,
    end_year: int,
):
    """Extract CHIRPS data."""

    input_dir = _no_ending_slash(input_dir)
    contours_df = gpd.read_file(contours_file)

    input_fs = storage.filesystem(input_dir)
    output_fs = storage.filesystem(output_file)

    contours = contours_df.to_crs("EPSG:4326")
    data = pd.DataFrame()

    for year in range(start_year, end_year + 1):

        for epi_week_dir in input_fs.glob(
            f"{input_dir}/{year}-*", re_add_protocol=True
        ):

            logger.info(f"Processing epi week {epi_week_dir.split('/')[-1]}")

            files = input_fs.glob(f"{epi_week_dir}/*.tif", re_add_protocol=True)

            # Skip incomplete epi weeks
            if len(files) < 6:
                logger.info(
                    f"Skipping incomplete epi week {epi_week_dir.split('/')[-1]}"
                )
                continue

            epi_year, epi_week = epi_week_dir.split("/")[-1].split("-")
            epi_array, epi_affine = extract_sum_epi_week(files)

            stats = zonal_stats(
                contours,
                epi_array,
                affine=epi_affine,
                nodata=np.nan,
                stats="sum count",
                geojson_out=True,
            )

            df = gpd.GeoDataFrame.from_features(stats)
            df = pd.DataFrame(df.drop(columns="geometry"))
            df["epi_year"], df["epi_week"], df["date"] = [
                epi_year,
                epi_week,
                f"{epi_week}/{epi_year}",
            ]

            data = pd.concat([data, df], axis=0, copy=False)

    data = data.reset_index(drop=True)

    output_fs.makedirs(output_fs.dirname(output_file), exist_ok=True)
    with output_fs.open(output_file, "w") as f:
        csv = data.to_csv(index=False)
        f.write(csv)
        logger.info(f"Aggregated statistics written to {output_file}.")


def _no_ending_slash(path):
    """Remove ending slash if needed."""
    if path.endswith("/"):
        return path[:-1]
    return path


if __name__ == "__main__":
    cli()
