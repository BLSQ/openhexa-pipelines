"""Download daily CHIRPS rain data for the African continent.
Process them based on the contours of the countries provided.
Results are aggregated by Epi week (CDC week).
"""
import enum
import gzip
import logging
import os
import string
import tempfile
import typing
from datetime import date, datetime, timedelta

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from rasterstats import zonal_stats
from s3fs import S3FileSystem
from sqlalchemy import create_engine

# comon is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError:
    # ignore import error -> work anyway (but define logging)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)


class ChirpsError(Exception):
    """CHIRPS base error."""


def filesystem(target_path: str) -> AbstractFileSystem:
    """Guess filesystem based on path"""

    client_kwargs = {}
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
        if target_protocol == "s3":
            fs_class = S3FileSystem
            client_kwargs = {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}
        elif target_protocol == "gcs":
            fs_class = GCSFileSystem
        elif target_protocol == "http" or target_protocol == "https":
            fs_class = HTTPFileSystem
        else:
            raise ValueError(f"Protocol {target_protocol} not supported.")
    else:
        fs_class = LocalFileSystem

    return fs_class(client_kwargs=client_kwargs)


@click.group()
def cli():
    """Download and process CHIRPS data."""
    pass


@cli.command()
@click.option("--output-dir", type=str, help="output directory", required=True)
@click.option("--start", type=str, help="start date", required=True)
@click.option("--end", type=str, help="end date", required=True)
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
@click.option("--debug", is_flag=True, default=False, help="enable debug logs")
def download(output_dir: str, start: int, end: int, overwrite: bool, debug: bool):
    """Download raw precipitation data."""
    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Downloading CHIRPS data from {start} to {end} into {output_dir}.")
    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()
    chirps = Chirps()
    chirps.download_range(start, end, output_dir, overwrite=overwrite)
    return


@cli.command()
@click.option("--start", type=str, help="start date", required=True)
@click.option("--end", type=str, help="end date", required=True)
@click.option("--contours", type=str, help="path to contours", required=True)
@click.option("--input-dir", type=str, help="chirps data directory", required=True)
@click.option("--output-dir", type=str, help="output data directory", required=True)
@click.option("--weekly-table", type=str, help="sql table name to weekly output")
@click.option("--monthly-table", type=str, help="sql table name to monthly output")
@click.option("--db-host", type=str, help="database hostname")
@click.option("--db-port", type=int, help="database port", default=5432)
@click.option("--db-name", type=str, help="database name")
@click.option("--db-user", type=str, help="database username")
@click.option("--db-password", type=str, help="database password")
@click.option("--debug", is_flag=True, default=False, help="enable debug logs")
def extract(
    start,
    end,
    contours,
    input_dir,
    output_dir,
    weekly_table,
    monthly_table,
    db_host,
    db_port,
    db_name,
    db_user,
    db_password,
    debug,
):
    """Compute zonal statistics."""

    if debug:
        logger.setLevel(logging.DEBUG)

    fs_output = filesystem(output_dir)
    fs_output.mkdirs(output_dir, exist_ok=True)
    weekly_csv = os.path.join(output_dir, "weekly.csv")
    monthly_csv = os.path.join(output_dir, "monthly.csv")

    logger.info(f"Computing zonal statistics for {contours}.")

    if contours.startswith("pg://"):
        con = create_engine(
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        table = contours.split("/")[-1]
        contours_data = gpd.read_postgis(f"SELECT * FROM {table}", con, "geometry")
    else:
        fs = filesystem(contours)
        with fs.open(contours) as f:
            contours_data = gpd.read_file(f)

    # Ignore invalid geometries
    contours_data = contours_data[contours_data.is_simple]
    contours_data = contours_data[contours_data.is_valid]

    if contours_data.crs and contours_data.crs != "EPSG:4326":
        contours_data = contours_data.to_crs("EPSG:4326")

    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()

    weekly_data = weekly_stats(
        contours=contours_data, start=start, end=end, chirps_dir=input_dir
    )

    # force some types:
    weekly_data["epi_week"] = weekly_data["epi_week"].astype(int)
    weekly_data["epi_year"] = weekly_data["epi_year"].astype(int)
    weekly_data["count"] = weekly_data["count"].astype(int)

    with fs_output.open(weekly_csv, "w") as f:
        weekly_data.to_csv(f)

    if weekly_table:
        con = create_engine(
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # remove potential SQL injection
        # we can't use params={} from read_sql because it escape the table name
        # which makes the query invalid
        table = "".join(
            [c for c in weekly_table if c in string.ascii_letters + string.digits + "_"]
        )

        try:
            # select the last data in the table
            max_year, max_week = pd.read_sql(
                """
                select
                    max(epi_year::int) as max_year,
                    max(epi_week::int) as max_week
                from "%(table)s"
                where (epi_year::int) = (
                    select max(epi_year::int) as max_year
                    from "%(table)s"
                )
            """
                % {"table": table},
                con,
            ).values[0]
        except Exception:
            max_year, max_week = None, None

        if max_year and max_week:
            weekly_data[
                (weekly_data.epi_year >= max_year) & (weekly_data.epi_week > max_week)
            ].to_sql(table, con, if_exists="append", index=False, chunksize=4096)
        else:
            weekly_data.to_sql(
                table, con, if_exists="append", index=False, chunksize=4096
            )

    monthly_data = monthly_stats(
        contours=contours_data, start=start, end=end, chirps_dir=input_dir
    )

    # force some types:
    monthly_data["epi_month"] = monthly_data["epi_month"].astype(int)
    monthly_data["epi_year"] = monthly_data["epi_year"].astype(int)
    monthly_data["count"] = monthly_data["count"].astype(int)

    with fs_output.open(monthly_csv, "w") as f:
        monthly_data.to_csv(f)

    if monthly_table:
        con = create_engine(
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # remove potential SQL injection
        # we can't use params={} from read_sql because it escape the table name
        # which makes the query invalid
        table = "".join(
            [
                c
                for c in monthly_table
                if c in string.ascii_letters + string.digits + "_"
            ]
        )

        try:
            # select the last data in the table
            max_year, max_month = pd.read_sql(
                """
                select
                    max(epi_year::int) as max_year,
                    max(epi_month::int) as max_month
                from "%(table)s"
                where (epi_year::int) = (
                    select max(epi_year::int) as max_year
                    from "%(table)s"
                )
            """
                % {"table": table},
                con,
            ).values[0]
        except Exception:
            max_year, max_month = None, None

        if max_year and max_month:
            monthly_data[
                (monthly_data.epi_year >= max_year)
                & (monthly_data.epi_month > max_month)
            ].to_sql(table, con, if_exists="append", index=False, chunksize=4096)
        else:
            monthly_data.to_sql(
                table, con, if_exists="append", index=False, chunksize=4096
            )

    return


class EpiSystem(enum.Enum):
    CDC = "CDC"
    WHO = "WHO"


class EpiWeek:
    """How do we define the first week of the year/January ?

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
    """

    def __init__(self, date_object: date, system: EpiSystem = "CDC"):
        self.from_date(date_object, system)

    def __iter__(self):
        """Iterate over days of the epi. week."""
        n_days = (self.end - self.start).days + 1
        for wday in range(0, n_days):
            yield self.start + timedelta(days=wday)

    def __eq__(self, other):
        """Compare only week and year."""
        return self.week == other.week and self.year == other.year

    def __repr__(self):
        return f"EpiWeek({self.year}W{self.week})"

    def from_date(self, date_object: date, system: EpiSystem = "CDC"):
        """Get epidemiological week info from date object."""
        systems = {"CDC": 0, "WHO": 1}
        if system.upper() not in systems.keys():
            raise ValueError("System not in {}".format(list(systems.keys())))

        week_day = (date_object - timedelta(days=systems[system.upper()])).isoweekday()
        week_day = 0 if week_day == 7 else week_day  # Week : Sun = 0 ; Sat = 6

        # Start the weekday on Sunday (CDC)
        self.start = date_object - timedelta(days=week_day)
        # End the weekday on Saturday (CDC)
        self.end = self.start + timedelta(days=6)

        week_start_year_day = self.start.timetuple().tm_yday
        week_end_year_day = self.end.timetuple().tm_yday

        if week_end_year_day in range(4, 11):
            self.week = 1
        else:
            self.week = ((week_start_year_day + 2) // 7) + 1

        if week_end_year_day in range(4, 11):
            self.year = self.end.year
        else:
            self.year = self.start.year


def epiweek_range(start: date, end: date) -> typing.List[EpiWeek]:
    """Get a range of epidemiological weeks between two dates."""
    epiweeks = []
    day = start
    while day <= end:
        epiweek = EpiWeek(day)
        if epiweek not in epiweeks:
            epiweeks.append(epiweek)
        day += timedelta(days=1)
    return epiweeks


def _compress(src_raster: str, dst_raster: str):
    """Compress a raster to save storage space."""
    with rasterio.open(src_raster) as src:
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
        with rasterio.open(dst_raster, "w", **dst_profile) as dst:
            dst.write(src.read(1), 1)
    return dst_raster


def _download(url: str, dst_file: str, timeout: int = 60):
    """Download file at URL."""
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dst_file, "wb") as f:
        if url.endswith(".gz"):
            f.write(gzip.decompress(r.content))
        else:
            f.write(r.content)
    return dst_file


class Chirps:
    def __init__(
        self, version: str = "2.0", zone: str = "africa", timely: str = "daily"
    ):
        self.version = version
        self.zone = zone
        self.timely = timely

    @property
    def base_url(self):
        return (
            "https://data.chc.ucsb.edu/products/CHIRPS-"
            f"{self.version}/{self.zone}_{self.timely}/tifs/p05"
        )

    def fname(self, day: date):
        """Get file name of daily CHIRPS data for a given date."""
        return f"chirps-v2.0.{day.year}.{day.month:0>2d}.{day.day:0>2d}.tif.gz"

    def download(
        self, day: date, output_dir: str, timeout: int = 60, overwrite: bool = False
    ):
        """Download a CHIRPS file identified by a date."""
        fs = filesystem(output_dir)
        fname = self.fname(day)
        output_file = os.path.join(output_dir, fname.replace(".gz", ""))

        if fs.exists(output_file) and not overwrite:
            logger.info(f"{fname} already exists. Skipping download.")
            return

        url = f"{self.base_url}/{day.year}/{fname}"
        logger.info(
            f"Downloading CHIRPS data for date {day.strftime('%Y-%m-%d')} {url}."
        )

        r = requests.head(url)
        if r.status_code != 200:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            raster = _download(
                url, os.path.join(tmp_dir, "raster.tif"), timeout=timeout
            )
            raster = _compress(raster, os.path.join(tmp_dir, "raster_compressed.tif"))
            fs.put(raster, output_file)
        logger.debug(f"Moved downloaded file to {output_file}.")

        return output_file

    def download_range(
        self, start: date, end: date, output_dir: str, overwrite: bool = False
    ):
        """Download all CHIRPS daily products to cover a given date range."""
        day = start
        while day <= end:
            per_year = os.path.join(output_dir, str(day.year))
            os.makedirs(per_year, exist_ok=True)
            self.download(day, output_dir=per_year, overwrite=overwrite)
            day += timedelta(days=1)


def chirps_path(data_dir: str, date_object: date):
    """Return CHIRPS raster file path for a given date."""
    fs = filesystem(data_dir)
    pattern = os.path.join(
        data_dir, str(date_object.year), f"*{date_object.strftime('%Y.%m.%d')}.tif"
    )
    search = fs.glob(pattern)
    if search:
        if "://" in data_dir:
            protocol = data_dir.split("://")[0]
            return f"{protocol}://{search[0]}"
        else:
            return search[0]
    return None


def raster_cumsum(
    rasters: typing.List[str], bounds: typing.Tuple[float]
) -> typing.Tuple[np.ndarray, Affine, float]:
    """Compute cumulative sum between rasters."""
    logger.debug(f"Computing cumulative sum for {len(rasters)} rasters.")
    fs = filesystem(rasters[0])
    # Get raster metadata from first raster in the list
    with fs.open(rasters[0]) as fp:
        with rasterio.open(fp) as src:
            xmin, ymin, xmax, ymax = bounds
            nodata = src.nodata
            window = rasterio.windows.from_bounds(
                xmin, ymin, xmax, ymax, transform=src.transform
            )
            affine = src.window_transform(window)
            height, width = src.read(1, window=window, masked=True).shape

    cumsum = np.zeros(shape=(height, width), dtype=np.float32)
    for rst in rasters:
        with fs.open(rst) as fp:
            with rasterio.open(fp) as src:
                data = src.read(1, window=window)
                has_data = data != nodata
                cumsum[has_data] += data[has_data]

    return cumsum, affine, nodata


def weekly_stats(
    contours: gpd.GeoDataFrame, start: date, end: date, chirps_dir: str
) -> pd.DataFrame:
    """Compute weekly precipitation aggregation statistics.

    Parameters
    ----------
    contours : GeoDataFrame
        Contours geodataframe with a geometry column (EPSG:4326).
    start : date
        Start date.
    end : date
        End date.
    chirps_dir : str
        CHIRPS data root directory.

    Return
    ------
    DataFrame
        Weekly stats as a dataframe of length n_weeks * n_contours.
    """
    fs = filesystem(chirps_dir)
    weeks = epiweek_range(start, end)

    dataframe = pd.DataFrame(columns=contours.columns)

    for week in weeks:

        # Get the list of all daily rasters for the epi week
        rasters = [chirps_path(chirps_dir, day) for day in week]
        if not all(rasters):
            logger.info(f"Epidemiological week {week} is incomplete. Skipping.")
            continue

        logger.info(f"Computing zonal statistics for epidemiological week {week}.")
        # Compute zonal statistics based on precipitation cumulative sum
        cumsum, affine, nodata = raster_cumsum(rasters, contours.total_bounds)

        shapes = [geom.__geo_interface__ for geom in contours.geometry]
        logger.debug(f"Computing zonal stats for {len(shapes)} polygons.")
        stats = zonal_stats(
            shapes,
            cumsum,
            affine=affine,
            nodata=nodata,
            stats=["sum", "count"],
        )

        logger.debug("Formatting output.")
        dataframe_week = contours.copy()
        dataframe_week["period"] = f"{week.year}W{week.week}"
        dataframe_week["epi_year"] = str(week.year)
        dataframe_week["epi_week"] = str(week.week)
        dataframe_week["start_date"] = week.start.strftime("%Y-%m-%d")
        dataframe_week["end_date"] = week.end.strftime("%Y-%m-%d")
        dataframe_week["mid_date"] = week.start + (week.end - week.start) / 2
        dataframe_week["count"] = [stat["count"] for stat in stats]
        dataframe_week["sum"] = [stat["sum"] for stat in stats]
        dataframe = dataframe.append(dataframe_week)

    if "geometry" in dataframe.columns:
        dataframe = dataframe.drop(columns=["geometry"])
    return dataframe


def _iter_month_days(year: int, month: int):
    """Iterate over days in a month."""
    start = date(year, month, 1)
    for d in range(0, 31):
        day = start + timedelta(days=d)
        if day.month == month:
            yield day


def monthly_stats(
    contours: gpd.GeoDataFrame, start: date, end: date, chirps_dir: str
) -> pd.DataFrame:
    """Compute weekly precipitation aggregation statistics.

    Parameters
    ----------
    contours : GeoDataFrame
        Contours geodataframe with a geometry column (EPSG:4326).
    start : date
        Start date.
    end : date
        End date.
    chirps_dir : str
        CHIRPS data root directory.

    Return
    ------
    DataFrame
        Weekly stats as a dataframe of length n_weeks * n_contours.
    """
    dataframe = pd.DataFrame(columns=contours.columns)

    year, month = start.year, start.month

    while year <= end.year and month <= end.month:

        days_in_month = [d for d in _iter_month_days(year, month)]

        # Get the list of all daily rasters for the month
        rasters = [chirps_path(chirps_dir, d) for d in days_in_month]

        if all(rasters):

            logger.info(f"Computing zonal statistics for month {year}{month:02}.")
            # Compute zonal statistics based on precipitation cumulative sum
            cumsum, affine, nodata = raster_cumsum(rasters, contours.total_bounds)
            stats = zonal_stats(
                [geom.__geo_interface__ for geom in contours.geometry],
                cumsum,
                affine=affine,
                nodata=nodata,
                stats=["sum", "count"],
            )

            dataframe_month = contours.copy()
            dataframe_month["period"] = f"{year}{month:02}"
            dataframe_month["epi_year"] = year
            dataframe_month["epi_month"] = int(month)
            dataframe_month["start_date"] = days_in_month[0].strftime("%Y-%m-%d")
            dataframe_month["end_date"] = days_in_month[-1].strftime("%Y-%m-%d")
            dataframe_month["mid_date"] = (
                days_in_month[0] + (days_in_month[-1] - days_in_month[0]) / 2
            )
            dataframe_month["count"] = [stat["count"] for stat in stats]
            dataframe_month["sum"] = [stat["sum"] for stat in stats]
            dataframe = dataframe.append(dataframe_month)

        else:

            logger.info(f"Month {year}{month:02} is incomplete. Skipping.")

        if month < 12:
            month += 1
        else:
            year += 1
            month = 1

    if "geometry" in dataframe.columns:
        dataframe = dataframe.drop(columns=["geometry"])
    return dataframe


if __name__ == "__main__":
    cli()
