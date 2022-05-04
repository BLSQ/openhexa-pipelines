"""Download and extract data from CPC Global Daily Temperarature.

See <https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html>.
"""

import enum
import logging
import os
import string
import tempfile
from datetime import date, datetime, timedelta
from typing import List, Tuple

import click
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
import xarray as xr
from appdirs import user_cache_dir
from rasterio.features import rasterize
from sqlalchemy import create_engine

# common is a script to set parameters on production
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

APP_NAME = "openhexa-pipelines-temperature"
APP_AUTHOR = "bluesquare"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--start", type=str, required=True, help="start date of period to sync")
@click.option("--end", type=str, required=True, help="end date of period to sync")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def sync(start: str, end: str, output_dir: str, overwrite: bool):
    """Sync yearly temperature data sets."""
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    fs = filesystem(output_dir)
    fs.makedirs(output_dir, exist_ok=True)
    for var in ("tmin", "tmax"):
        sync_data(
            data_dir=output_dir, start_year=start.year, end_year=end.year, variable=var
        )


@cli.command()
@click.option("--start", type=str, required=True, help="start date of analysis")
@click.option("--end", type=str, required=True, help="end date of analysis")
@click.option("--boundaries", type=str, required=True, help="aggregation boundaries")
@click.option(
    "--boundaries-name", type=str, required=True, help="column with boundary name"
)
@click.option("--input-dir", "-i", type=str, required=True, help="input data directory")
@click.option("--output-file", "-o", type=str, required=True, help="output dataset")
@click.option("--overwrite", is_flag=True, default=False, help="overwrite files")
def daily(
    start: str,
    end: str,
    boundaries: str,
    boundaries_name: str,
    input_dir: str,
    output_file: str,
    overwrite: bool,
):
    """Compute daily zonal statistics."""
    fs = filesystem(boundaries)
    with fs.open(boundaries) as f:
        boundaries = gpd.read_file(f, driver="GPKG")
    boundaries = fix_geometries(boundaries)

    # make sure boundaries are in lat/lon coordinates
    dst_crs = pyproj.CRS.from_epsg(4326)
    if not boundaries.crs:
        boundaries.crs = dst_crs
    if boundaries.crs != dst_crs:
        boundaries = boundaries.to_crs(dst_crs)

    boundaries.set_index(boundaries_name, inplace=True)
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    daily_stats = daily_zonal_statistics(
        data_dir=input_dir,
        geoms=boundaries.geometry,
        start=start,
        end=end,
        all_touched=True,
    )

    fs = filesystem(output_file)
    if fs.exists(output_file):
        if overwrite:
            fs.rm(output_file)
        else:
            raise FileExistsError(f"{output_file} already exists.")

    # xarray supports writing to File-like objects only with the scipy engine.
    # we want the h5netcdf engine so we create a temporary file to be able to
    # provide a file path to xr.dataset.to_netcdf()
    fs.makedirs(os.path.dirname(output_file), exist_ok=True)
    with tempfile.NamedTemporaryFile() as tmp_file:
        daily_stats.to_netcdf(tmp_file.name, engine="h5netcdf")
        fs.put(tmp_file.name, output_file)
        logger.info(f"Daily zonal stats saved into {output_file}.")


@cli.command()
@click.option("--daily-file", type=str, required=True, help="daily zonal statistics")
@click.option(
    "--frequency",
    type=click.Choice(["weekly", "monthly"]),
    required=True,
    help="aggregation frequency",
)
@click.option("--output-file-tmin", type=str, required=False, help="output csv file")
@click.option("--output-file-tmax", type=str, required=False, help="output csv file")
@click.option("--output-table-tmin", type=str, required=False, help="output SQL table")
@click.option("--output-table-tmax", type=str, required=False, help="output SQL table")
@click.option("--db-user", type=str, required=False, help="DB username")
@click.option("--db-password", type=str, required=False, help="DB password")
@click.option("--db-host", type=str, required=False, help="DB hostname")
@click.option("--db-port", type=int, required=False, help="DB port")
@click.option("--db-name", type=str, required=False, help="DB name")
def aggregate(
    daily_file: str,
    frequency: str,
    output_file_tmin: str,
    output_file_tmax: str,
    output_table_tmin: str,
    output_table_tmax: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: str,
    db_name: str,
):
    """Compute weekly zonal statistics."""
    fs_in = filesystem(daily_file)

    for variable, output_file, output_table in zip(
        ("tmin", "tmax"),
        (output_file_tmin, output_file_tmax),
        (output_table_tmin, output_table_tmax),
    ):

        with tempfile.NamedTemporaryFile() as tmp_file:
            fs_in.get(daily_file, tmp_file.name)
            daily = xr.open_dataset(tmp_file.name, engine="h5netcdf").load()
            if frequency == "weekly":
                stats = weekly_zonal_stats(daily, variable)
            elif frequency == "monthly":
                stats = monthly_zonal_stats(daily, variable)
            else:
                raise ValueError("Unrecognized frequency option.")
            daily.close()

        if output_file:
            fs_out = filesystem(output_file)
            with fs_out.open(output_file, "w") as f:
                stats.to_csv(f, index=False)
            logger.info(f"Saved weekly zonal statistics to {output_file}.")

        if output_table:

            con = create_engine(
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )

            # remove potential SQL injection
            # we can't use params={} from read_sql because it escape the table name
            # which makes the query invalid
            table = "".join(
                [
                    c
                    for c in output_table
                    if c in string.ascii_letters + string.digits + "_"
                ]
            )
            stats.to_sql(table, con, if_exists="replace", index=False, chunksize=4096)
            logger.info(f"Exported weekly zonal statistics to {output_table}.")


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

    def __str__(self):
        return f"{self.year}W{str(self.week).zfill(2)}"

    def __hash__(self):
        return id(self)

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


def epiweek_range(start: date, end: date) -> List[EpiWeek]:
    """Get a range of epidemiological weeks between two dates."""
    epiweeks = []
    day = start
    while day <= end:
        epiweek = EpiWeek(day)
        if epiweek not in epiweeks:
            epiweeks.append(epiweek)
        day += timedelta(days=1)
    return epiweeks


def filesystem(target_path: str, cache: bool = False) -> fsspec.AbstractFileSystem:
    """Guess filesystem based on path.

    Parameters
    ----------
    target_path : str
        Target file path starting with protocol.
    cache : bool, optional
        Cache remote file locally (default=False).

    Return
    ------
    AbstractFileSystem
        Local, S3, or GCS filesystem. WholeFileCacheFileSystem if
        cache=True.
    """
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
    else:
        target_protocol = "file"

    if target_protocol not in ("file", "s3", "gcs"):
        raise ValueError(f"Protocol {target_protocol} not supported.")

    client_kwargs = {}
    if target_protocol == "s3":
        client_kwargs = {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}

    if cache:
        return fsspec.filesystem(
            protocol="filecache",
            target_protocol=target_protocol,
            target_options={"client_kwargs": client_kwargs},
            cache_storage=user_cache_dir(appname=APP_NAME, appauthor=APP_AUTHOR),
        )
    else:
        return fsspec.filesystem(protocol=target_protocol, client_kwargs=client_kwargs)


def fix_geometries(geodf: gpd.GeoDataFrame):
    """Try to fix invalid geometries in a geodataframe.

    Parameters
    ----------
    geodf : geodataframe
        Input geodataframe with a geometry column.

    Return
    ------
    geodf : geodataframe
        Updated geodataframe with valid geometries.
    """
    geodf_ = geodf.copy()
    n_features_orig = len(geodf_)
    for i, row in geodf_.iterrows():
        if not row.geometry.is_valid:
            geodf_.at[i, "geometry"] = row.geometry.buffer(0)
    geodf_ = geodf_[geodf_.is_simple]
    geodf_ = geodf_[geodf_.is_valid]
    n_features = len(geodf_)
    if n_features < n_features_orig:
        logger.warn(
            f"{n_features_orig - n_features} are invalid and were excluded from the analysis."
        )
    return geodf_


BASE_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp"


def get_download_url(year: int, variable: str = "tmin") -> str:
    """Get download URL for the Global Daily CPC yearly product."""
    return f"{BASE_URL}/{variable}.{year}.nc"


def download(url: str, dst_fp: str, timeout: int = 30, overwrite: bool = False) -> str:
    """Download remote file."""
    fs = filesystem(dst_fp)
    if fs.exists(dst_fp) and not overwrite:
        raise FileExistsError(f"{dst_fp} already exists.")
    fs.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    with tempfile.NamedTemporaryFile() as tmp_file:
        with requests.get(url, stream=True, timeout=timeout) as r:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
        fs.put(tmp_file.name, dst_fp)
    logger.info(f"Downloaded {url} into {dst_fp}.")
    return dst_fp


def sync_data(
    data_dir: str,
    start_year: int,
    end_year: int,
    variable: str = "tmin",
):
    """Download required source data.

    Remote file is downloaded if :
        * No file is available in the data dir
        * File in data dir has a different size (in bytes)

    If error 404, then the product is considered as not available.

    Parameters
    ----------
    data_dir : str
        Directory where raw data are stored.
    start_year : int
        Start sync year.
    end_year : int
        End sync year.
    variable : str, optional
        "tmin" or "tmax".
    """
    for year in range(start_year, end_year + 1):

        url = get_download_url(year, variable)
        fp = os.path.join(data_dir, url.split("/")[-1])
        r = requests.head(url)

        if r.status_code == 404:
            logger.info(f"Product not available at {url}.")
            continue

        size = int(r.headers.get("content-length"))
        fs = filesystem(fp)

        if fs.exists(fp):
            if fs.size(fp) == size:
                logger.info(f"{fp} exists and is up-to-date.")
            else:
                logger.info(f"{fp} exists but is not up-to-date.")
                download(url, fp, overwrite=True)
        else:
            logger.info(f"{fp} does not exist.")
            download(url, fp, overwrite=False)

    return data_dir


def get_date_range(data_dir: str) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Get min and max date available in raw files found in a directory."""
    fs = filesystem(data_dir)
    min_year = 3000
    max_year = 0
    for fp in fs.glob(os.path.join(data_dir, "*.nc")):
        year = int(os.path.basename(fp).split(".")[1])
        if year < min_year:
            min_year = year
            min_year_fp = fp
        if year > max_year:
            max_year = year
            max_year_fp = fp

    with fs.open(min_year_fp, "rb") as f:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            min_date = ds.time.values.min()

    with fs.open(max_year_fp, "rb") as f:
        with xr.open_dataset(f, engine="h5netcdf") as ds:
            max_date = ds.time.values.max()

    return pd.to_datetime(min_date), pd.to_datetime(max_date)


def get_yearly_data(data_dir: str, year: int) -> np.ndarray:
    """Get yearly data for both tmin and tmax.

    Return an array of shape (n_vars, n_days, height, width) with n_vars=2 (tmin and tmax).
    """
    dst_arrays = []
    fs = filesystem(data_dir)
    for var in ("tmin", "tmax"):
        fp = os.path.join(data_dir, f"{var}.{year}.nc")
        with tempfile.NamedTemporaryFile() as tmp_file:
            fs.get(fp, tmp_file.name)
            src_ds = xr.open_dataarray(tmp_file.name)
            dst_arrays.append(src_ds.values.astype(np.float64).copy())
            src_ds.close()
        logger.info(f"Loaded measurements from file {fp}.")
    dst_ndarray = np.asarray(dst_arrays, dtype=np.float64)
    logger.info(
        f"Converted measurement values to a ndarray of shape {dst_ndarray.shape} and size {round((dst_ndarray.size * dst_ndarray.itemsize) / 1024 * 1024), 2}MB."
    )
    return dst_ndarray


def rotate_raster(data: np.ndarray) -> np.ndarray:
    """Reproject raster data to WGS 84 grid.

    Reproject raster grid from 0-360 to -180-180 latitude by applying a
    rotation around the Greenwich meridian. Output raster has the same
    shape.
    """
    if not data.shape == (360, 720):
        raise ValueError(f"Invalid raster shape: {data.shape}.")
    arr = np.empty_like(data)
    arr[0:360, :360] = data[0:360, 360:]
    arr[0:360, 360:] = data[0:360, :360]
    return arr


def daily_zonal_statistics(
    data_dir: str,
    geoms: gpd.GeoSeries,
    start: datetime,
    end: datetime,
    all_touched: bool = True,
) -> np.ndarray:
    """Compute daily zonal statistics for each input geometry.

    Output temperature value associated with a geometry is the mean value of all
    the intersecting pixels -- regardless of the surface of the intersection.

    Parameters
    ----------
    data_dir : str
        Input data directory (with raw .nc files).
    geoms : GeoSeries
        Input aggregation areas as shapely geometries.
    start : datetime
        Beginning of data extraction.
    end : datetime
        End of data extraction.
    all_touched : bool, optional
        Rasterize all intersecting cells (default=True).

    Return
    ------
    xarray dataset
        Dataset with daily aggregated data of shape (n_geoms, n_days).
    """
    # do not process dates without any data available
    min_date, max_date = get_date_range(data_dir)
    start = max(start, min_date)
    end = min(end, max_date)
    logger.info(f"Start date {str(start)}")
    logger.info(f"End date {str(end)}")

    # world wgs 84
    transform = rasterio.Affine(0.5, 0.0, -180, 0.0, -0.5, 90.0)

    # rasterize input geometries as binary raster layers. nb: input geometries
    # cannot be rasterized into the same layer as some vector layers will
    # overlap when rasterized if all_touched is set to True
    areas = np.empty(shape=(len(geoms), 360, 720), dtype=np.bool_)
    logger.info(f"Rasterizing masks for {len(areas)} geometries.")
    for i, geom in enumerate(geoms):
        area = rasterize(
            [geom],
            out_shape=(360, 720),
            fill=0,
            default_value=1,
            transform=transform,
            all_touched=all_touched,
            dtype="uint8",
        )
        area = rotate_raster(area)  # align to temperature raster grid
        if np.count_nonzero(area) == 0:
            logger.warn(f"No cell covered by input geometry (index={i}).")
        areas[i, :, :] = area == 1

    data = {"tmin": [], "tmax": []}
    drange = [day for day in pd.date_range(start, end)]

    year = start.year
    measurements = get_yearly_data(data_dir, year)
    logger.info(f"Loaded data for year {year}.")

    for day in drange:

        logger.info(f"Computing zonal stats for {day.strftime('%Y-%m-%d')}.")

        # each new year, reload measurements
        if day.year != year:
            year = day.year
            measurements = get_yearly_data(data_dir, year)
            logger.info(f"Loaded data for year {year}.")

        for i, var in enumerate(("tmin", "tmax")):
            # temporal index of measurements in yearly data is equal to julian day
            jday = day.toordinal() - datetime(day.year, 1, 1).toordinal()
            measurements_day = measurements[i, jday, :, :]

            means = []
            for j in range(0, len(geoms)):
                # get the subset of the measurements which are covered by the
                # rasterized input geometry
                measurements_area = measurements_day[areas[j, :, :]]
                means.append(measurements_area[~np.isnan(measurements_area)].mean())
            data[var].append(means)

    logger.info(f"Extracted daily zonal statistics for {len(drange)} days.")
    # return result as a structured 3d xarray dataset of shape (2, n_areas, n_days)
    ds = xr.Dataset(
        data_vars={
            "tmin": (["area", "time"], np.array(data["tmin"]).T),
            "tmax": (["area", "time"], np.array(data["tmax"]).T),
        },
        coords={"area": geoms.index.values, "time": pd.date_range(start, end)},
    )

    return ds


def weekly_zonal_stats(dataset: xr.Dataset, variable: str) -> pd.DataFrame:
    """Weekly aggregate from daily temperature data.

    Daily temperature values are aggregated using three different functions:
    mean, min, and max. Weeks are epidemiological weeks.

    Parameters
    ----------
    dataset : xarray dataset
        Daily zonal statistics.
    variable : str
        "tmin" or "tmax".

    Return
    ------
    dataframe
        Weekly zonal statistics in long format (nrows = n_epiweeks * n_boundaries).
    """
    df = pd.DataFrame(
        data=dataset[variable].values.T.ravel(),
        index=pd.MultiIndex.from_product(
            (dataset.time.values, dataset.area.values), names=("date", "name")
        ),
        columns=[variable],
    ).reset_index()

    df["epiweek"] = df.date.apply(lambda x: EpiWeek(x))
    df["year"] = df.epiweek.apply(lambda x: x.year)
    df["week"] = df.epiweek.apply(lambda x: x.week)
    df["start"] = df.epiweek.apply(lambda x: x.start)
    df["end"] = df.epiweek.apply(lambda x: x.end)
    df["epiweek"] = df.epiweek.astype(str)

    by_period = df.groupby(by=["epiweek", "year", "week", "start", "end", "name"])
    agg_mean = by_period.mean().rename(columns={variable: f"{variable}_mean"})
    agg_min = by_period.min().rename(columns={variable: f"{variable}_min"})
    agg_max = by_period.max().rename(columns={variable: f"{variable}_max"})
    agg = pd.merge(agg_mean, agg_min, left_index=True, right_index=True)
    agg = pd.merge(agg, agg_max, left_index=True, right_index=True)
    agg = agg.reset_index()

    agg["mid"] = agg.start + (agg.end - agg.start) / 2

    for column in agg.columns:
        if agg[column].dtype == "float32":
            agg[column] = agg[column].astype(float).round(1)

    return agg[
        [
            "epiweek",
            "name",
            "year",
            "week",
            "start",
            "end",
            f"{variable}_mean",
            f"{variable}_min",
            f"{variable}_max",
        ]
    ]


def monthly_zonal_stats(dataset: xr.Dataset, variable: str) -> pd.DataFrame:
    """Monthly aggregate from daily temperature data.

    Daily temperature values are aggregated using three different functions:
    mean, min, and max.

    Parameters
    ----------
    dataset : xarray dataset
        Daily zonal statistics.
    variable : str
        "tmin" or "tmax".

    Return
    ------
    dataframe
        Monthly zonal statistics in long format (nrows = n_months * n_boundaries).
    """
    df = pd.DataFrame(
        data=dataset[variable].values.T.ravel(),
        index=pd.MultiIndex.from_product(
            (dataset.time.values, dataset.area.values), names=("date", "name")
        ),
        columns=[variable],
    ).reset_index()

    df["month"] = df.date.apply(lambda x: x.month)
    df["year"] = df.date.apply(lambda x: x.year)

    by_period = df.groupby(by=["month", "year", "name"])
    agg_mean = by_period.mean().rename(columns={variable: f"{variable}_mean"})
    agg_min = by_period.min().rename(columns={variable: f"{variable}_min"})
    agg_max = by_period.max().rename(columns={variable: f"{variable}_max"})
    agg = pd.merge(agg_mean, agg_min, left_index=True, right_index=True)
    agg = pd.merge(agg, agg_max, left_index=True, right_index=True)
    agg = agg.reset_index()

    agg = agg.rename(columns={"date_x": "start", "date_y": "end"})
    agg["mid"] = agg.start + (agg.end - agg.start) / 2
    agg["period"] = agg.start.apply(lambda d: d.strftime("%Y%m"))

    for column in agg.columns:
        if agg[column].dtype == "float32":
            agg[column] = agg[column].astype(float).round(1)

    return agg[
        [
            "period",
            "name",
            "year",
            "month",
            "start",
            "end",
            f"{variable}_mean",
            f"{variable}_min",
            f"{variable}_max",
        ]
    ]


if __name__ == "__main__":
    cli()
