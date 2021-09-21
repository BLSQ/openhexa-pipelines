"""Download daily CHIRPS rain data for the African continent.
Process them based on the contours of the countries provided.
Results are aggregated by Epi week (CDC week).
"""

import os
import gzip
from datetime import date, timedelta
import logging

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterstats import zonal_stats
import s3fs

CHIRPS_VERSION = "2.0"
CHIRPS_TIMELY = "daily"
CHIRPS_ZONE = "africa"
# Year folder added on the fly by the download_chirps_daily function.
CHIRPS_URL = f"https://data.chc.ucsb.edu/products/CHIRPS-{CHIRPS_VERSION}/{CHIRPS_ZONE}_{CHIRPS_TIMELY}/tifs/p05/"
CHIRPS_BASENAME = (
    "chirps-v2.0.{chirps_year}.{chirps_month:0>2d}.{chirps_day:0>2d}.tif.gz"
)

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
    pass


@cli.command()
@click.option("--start", type=int, help="start year")
@click.option("--end", type=int, help="end year")
@click.option("--output-dir", type=str, help="output directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def download(start, end, output_dir, overwrite):
    """Download raw precipitation data."""
    logger.info(f"Downloading CHIRPS data from {start} to {end} into {output_dir}.")

    if "AWS_S3_ENDPOINT" in os.environ:
        fs = s3fs.S3FileSystem(
            client_kwargs={
                "endpoint_url": f"http://{os.environ.get('AWS_S3_ENDPOINT')}"
            }
        )
    else:
        fs = s3fs.S3FileSystem()

    output_dir = _no_ending_slash(output_dir)

    if not output_dir.startswith("s3://"):
        raise ValueError(f"{output_dir} is not a valid S3 path.")

    download_chirps_daily(fs, output_dir, start, end, overwrite)


@cli.command()
@click.option("--start", type=int, help="start year")
@click.option("--end", type=int, help="end year")
@click.option("--contours", type=str, help="path to contours file")
@click.option("--input-dir", type=str, help="raw CHIRPS data directory")
@click.option("--output-file", type=str, help="output directory")
def extract(start, end, contours, input_dir, output_file):
    """Compute zonal statistics."""
    input_dir = _no_ending_slash(input_dir)
    contours_df = gpd.read_file(contours)
    logger.info(f"Computing zonal statistics for {len(contours_df)} areas.")

    if not input_dir.startswith("s3://"):
        raise ValueError(f"{input_dir} is not a valid S3 path.")
    if not output_file.startswith("s3://"):
        raise ValueError(f"{output_file} is not a valid S3 path.")

    if "AWS_S3_ENDPOINT" in os.environ:
        fs = s3fs.S3FileSystem(
            client_kwargs={
                "endpoint_url": f"http://{os.environ.get('AWS_S3_ENDPOINT')}"
            }
        )
    else:
        fs = s3fs.S3FileSystem()

    stats = extract_chirps_data(fs, contours_df, input_dir, start, end)
    with fs.open(output_file, "w") as f:
        csv = stats.to_csv(index=False)
        f.write(csv)
        logger.info(f"Aggregated statistics written to {output_file}.")


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


def download_chirps_data(fs, url, output_path):
    """Download .tif or .gzip file from URL.

    Parameters
    ----------
    fs : s3fs.FileSystem
        Instance of a remote filesystem.
    url : str
        Download URL.
    output_path : str
        S3 output path.
    """
    logger.info(f"Downloading {url}")
    tiff = False
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            # If 404, check if the `tif` file is available instead of `gzip`
            response = requests.get(url[:-3], stream=True, timeout=60)
            tiff = True

        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("HTTP exception error: {}".format(err))
        return
    except requests.exceptions.RequestException as e:
        print("Exception error {}".format(e))
        return

    with fs.open(output_path, "wb") as f:
        if tiff:
            f.write(response.content)
        else:
            f.write(gzip.decompress(response.content))


def download_chirps_daily(
    fs,
    output_dir,
    year_start,
    year_end,
    overwrite=False,
):
    """Download CHIRPS daily products for a given time range.

    Parameters
    ----------
    fs : s3fs.FileSystem
        Instance of a remote filesystem.
    output_dir : str
        CHIRPS download folder.
    year_start : int
        Start year.
    year_end : int
        End year.
    overwrite : bool, optional
        Overwrite existing files (default=False).
    """
    time_range = provide_time_range(year_start, year_end)
    time_serie = time_range.to_series().apply(
        lambda r: provide_epi_week(r.year, r.month, r.day)
    )

    for day, (epi_week, year, _, _) in time_serie.items():
        output_file = CHIRPS_BASENAME.format(
            chirps_year=day.year, chirps_month=day.month, chirps_day=day.day
        )
        output_path = f"{output_dir}/{year}-{epi_week:0>2d}/{output_file[:-3]}"
        url = f"{CHIRPS_URL}{day.year}/{output_file}"

        if fs.exists(output_path):
            size_file = fs.du(output_path)
            if size_file == 0:
                logger.info(f"Redownloading {output_path}")
                # Redownload incorrect files
                download_chirps_data(fs, url, output_path)
            if not overwrite:
                logger.info(f"Skipping {output_path}")
                continue  # Skip download if overwrite is False

        download_chirps_data(fs, url, output_path)


def rio_read_file(file, band=1, nodata=-9999):
    """Read a raster band.

    Parameters
    ----------
    file : str
        Path to raster file.
    band : int
        Band number.
    nodata : int or float, optional
        Nodata value (default = -9999).

    Returns
    -------
    ndarray
        Band data with assigned nodata values.
    """
    with rasterio.open(file) as src:
        ds = src.read(band)
    return np.where(ds == nodata, np.nan, ds)


def rio_get_affine(file):
    """Get Affine transform for use with zonal_stats."""
    with rasterio.open(file) as src:
        return src.transform


def extract_sum_epi_week(files):
    """Extract and sum the list of files.

    Return summed array and affine transform.
    """
    array_rain = [rio_read_file(f"s3://{file}") for file in files]
    array_sum = np.nansum(array_rain, axis=0)
    affine = rio_get_affine(f"s3://{files[0]}")
    return (array_sum, affine)


def extract_chirps_data(fs, contours, input_dir, year_start, year_end):
    """Extract CHIRPS data.

    Parameters
    ----------
    fs : s3fs.FileSystem
        Instance of a remote filesystem.
    contours : GeoDataFrame
        Aggregation areas.
    input_dir : str
        Location of raw CHIRPS data.
    year_start : int
        Start year.
    year_end : int
        End year.

    Returns
    -------
    data : GeoDataFrame
        Extracted CHIRPS data.
    """
    contours = contours.to_crs("EPSG:4326")
    data = pd.DataFrame()

    for year in range(year_start, year_end + 1):

        for epi_week_dir in fs.glob(f"{input_dir}/{year}-*"):

            logger.info(f"Processing epi week {epi_week_dir.split('/')[-1]}")

            files = fs.glob(f"{epi_week_dir}/*.tif")

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

    return data.reset_index(drop=True)


def _no_ending_slash(path):
    """Remove ending slash if needed."""
    if path.endswith("/"):
        return path[:-1]
    return path


if __name__ == "__main__":
    cli()