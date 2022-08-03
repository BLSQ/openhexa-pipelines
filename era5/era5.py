import enum
import logging
import os
import string
import tempfile
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Callable, List, Tuple

import cdsapi
import click
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from appdirs import user_cache_dir
from dateutil.relativedelta import relativedelta
from rasterio.features import rasterize
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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

APP_NAME = "openhexa-pipelines-era5"
APP_AUTHOR = "bluesquare"


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--start", type=str, required=True, help="start date of period to process"
)
@click.option("--end", type=str, required=True, help="end date of period to process")
@click.option("--boundaries", type=str, required=True, help="aggregation boundaries")
@click.option(
    "--column-name", type=str, required=True, help="column name for boundary name"
)
@click.option(
    "--column-id", type=str, required=True, help="column name for boundary unique ID"
)
@click.option(
    "--cds-variable", type=str, required=True, help="CDS variable of interest"
)
@click.option(
    "--agg-function",
    type=str,
    required=False,
    default="mean",
    help="spatial aggregation function",
)
@click.option(
    "--hours",
    type=str,
    required=False,
    default="12:00",
    help="hour of the day (or ALL)",
)
@click.option("--csv", type=str, required=False, help="output CSV file")
@click.option("--db-user", type=str, required=False, help="database username")
@click.option("--db-password", type=str, required=False, help="database password")
@click.option("--db-host", type=str, required=False, help="database hostname")
@click.option("--db-port", type=int, required=False, help="database port")
@click.option("--db-name", type=str, required=False, help="database name")
@click.option("--db_table", type=str, required=False, help="database table")
@click.option(
    "--cds-api-key",
    type=str,
    required=True,
    envvar="CDS_API_KEY",
    help="CDS api key",
)
@click.option(
    "--cds-api-uid",
    type=str,
    required=True,
    envvar="CDS_API_UID",
    help="CDS user ID",
)
@click.option("--cache-dir", type=str, required=False, help="cache data directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing data"
)
def aggregate(
    start: str,
    end: str,
    boundaries: str,
    column_name: str,
    column_id: str,
    cds_variable: str,
    agg_function: str,
    hours: str,
    csv: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_table: str,
    cds_api_key: str,
    cds_api_uid: str,
    cache_dir: str,
    overwrite: bool,
):
    # make sure that at least 1 output option has been correctly set, i.e. db
    # table and/or csv file
    db_params = [db_user, db_password, db_host, db_port, db_name, db_table]
    if any(db_params) and not all(db_params):
        raise ValueError("All database parameters must be specified")
    if not csv and not all(db_params):
        raise ValueError("No output specified")

    # load boundaries data and perform basic quality checks
    # fix geometries if needed
    fs = filesystem(boundaries)
    with tempfile.NamedTemporaryFile(suffix=".gpkg") as tmp_file:
        fs.get(boundaries, tmp_file.name)
        boundaries_data = gpd.read_file(tmp_file.name)
    if column_id not in boundaries_data.columns:
        raise ValueError(f"Column {column_id} not found")
    if column_name not in boundaries_data.columns:
        raise ValueError(f"Column {column_name} not found")
    boundaries_data = fix_geometries(boundaries_data)
    if not boundaries_data.crs:
        boundaries_data.crs = "EPSG:4326"
    if boundaries_data.crs != "EPSG:4326":
        boundaries_data = boundaries_data.to_crs("EPSG:4326")

    # spatial aggregation function
    if agg_function.lower() == "mean":
        agg_function = np.mean
    elif agg_function.lower() == "median":
        agg_function = np.median
    elif agg_function.sum() == "sum":
        agg_function = np.sum
    else:
        raise ValueError(f"Aggregation function {agg_function} not supported")

    # hour of the day
    if hours.lower() == "all":
        hours = "ALL"
    else:
        hours = hours.split(",")

    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    # find latest processed date in csv and/or postgres table
    # only later dates will be processed except if overwrite = False
    fs = filesystem(csv)
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
        csv_mode = "w"  # write mode
        if fs.exists(csv):
            if overwrite:
                fs.remove(csv)
            else:
                fs.get(csv, tmp_file.name)
                max_date = _max_date_csv(tmp_file.name)
                if max_date > start:
                    start = max_date + relativedelta(months=1)
                    csv_mode = "a"  # append mode

    bounds = (40.0, -22.0, -36.0, 62.0)  # Africa
    era = Era5(variable=cds_variable, bounds=bounds, cache_dir=cache_dir)
    era.init_cdsapi(api_key=cds_api_key, api_uid=cds_api_uid)

    with tempfile.TemporaryDirectory() as tmp_dir:

        extracts = []
        date_ = start
        while date_ <= end:
            fname = f"{cds_variable}_{date_.year:04}{date_.month:02}.nc"
            try:
                datafile = era.download(
                    date_.year, date_.month, hours, os.path.join(tmp_dir, fname)
                )
            except Era5MissingData:
                logger.info(f"Missing data for period {date_.year:04}{date_.month:02}")
                return
            extracts.append(
                zonal_stats(
                    boundaries=boundaries_data,
                    datafile=datafile,
                    agg_function=agg_function,
                    column_id=column_id,
                    column_name=column_name,
                )
            )
            date_ = date_ + relativedelta(months=1)
        df = pd.concat(extracts, ignore_index=True)

        # temperature from K to C
        if "temperature" in cds_variable:
            df.value = df.value - 273.15

        # precipitation m to mm
        if cds_variable == "total_precipitation":
            df.value = df.value * 1000

        if csv:
            tmp_file = os.path.join(tmp_dir, "extract.csv")
            df.to_csv(tmp_file, index=False, mode=csv_mode)
            fs.put(tmp_file, csv)
        if db_table:
            db_table_safe = _safe_from_injection(db_table)
            con = create_engine(
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            df.to_sql(
                db_table_safe,
                con,
                index=False,
                if_exists="replace" if overwrite else "append",
            )


@cli.command()
@click.option("--src-file", type=str, required=True, help="source daily data")
@click.option(
    "--agg-function", type=str, default="mean", help="function for temporal aggregation"
)
@click.option("--csv", type=str, required=False, help="output csv file")
@click.option("--db-user", type=str, required=False, help="database username")
@click.option("--db-password", type=str, required=False, help="database password")
@click.option("--db-host", type=str, required=False, help="database hostname")
@click.option("--db-port", type=int, required=False, help="database port")
@click.option("--db-name", type=str, required=False, help="database name")
@click.option("--db_table", type=str, required=False, help="database table")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing data"
)
def weekly(
    src_file: str,
    agg_function: str,
    csv: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_table: str,
    overwrite: bool,
):
    # make sure that at least 1 output option has been correctly set, i.e. db
    # table and/or csv file
    db_params = [db_user, db_password, db_host, db_port, db_name, db_table]
    if any(db_params) and not all(db_params):
        raise ValueError("All database parameters must be specified")
    if not csv and not all(db_params):
        raise ValueError("No output specified")

    # temporal aggregation function
    if agg_function.lower() == "mean":
        agg_function = np.mean
    elif agg_function.lower() == "median":
        agg_function = np.median
    elif agg_function.lower() == "sum":
        agg_function = np.sum
    else:
        raise ValueError(f"Aggregation function {agg_function} not supported")

    fs = filesystem(src_file)

    with tempfile.TemporaryDirectory() as tmp_dir:

        tmp_file = os.path.join(tmp_dir, os.path.basename(src_file))
        fs.put(src_file, tmp_file)
        logger.info(f"Using daily data from {src_file}")

        daily = pd.read_csv(
            tmp_file, parse_dates=[2]
        )  # third column should be "period"
        logger.info(f"Loaded {len(daily.period.unique())} days of data")

        daily["epi_week"] = daily.period.apply(lambda day: str(EpiWeek(day)))
        grouped = daily.groupby(by=["id", "epi_week"])

        def _agg_function(values):
            """Same as agg function but returns NA if the number of values is less
            than 7 (week incomplete)."""
            if len(values) < 7:
                return pd.NA
            return agg_function(values)

        weekly = grouped.aggregate(
            {"id": "first", "name": "first", "period": "first", "value": _agg_function}
        )
        weekly = weekly.drop(columns=["id"]).reset_index()

        # drop rows with nodata values
        orig_length = len(weekly)
        weekly = weekly.dropna()
        logger.info(f"Dropped {orig_length - len(weekly)} rows with nodata values")

        # reorder columns
        weekly = weekly[["id", "name", "epi_week", "value"]]

        if csv:
            tmp_file = os.path.join(tmp_dir, "extract.csv")
            weekly.to_csv(tmp_file, index=False)
            fs.put(tmp_file, csv)
            logger.info(f"Written CSV output into {csv}")
        if db_table:
            db_table_safe = _safe_from_injection(db_table)
            con = create_engine(
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            weekly.to_sql(
                db_table_safe,
                con,
                index=False,
                if_exists="replace",
            )
            logger.info(f"Written DB output into {db_table} ({db_name})")


class Era5MissingData(Exception):
    "Data is incomplete for the specified month"
    pass


def _safe_from_injection(db_table: str) -> str:
    """Remove potential SQL injection."""
    return "".join(
        [c for c in db_table if c in string.ascii_letters + string.digits + "_"]
    )


def _max_date_csv(src_file: str) -> datetime:
    """Get max processed date in csv file."""
    data = pd.read_csv(src_file)
    max_date = pd.to_datetime(data.period).max().to_pydatetime()
    return max_date


def _max_date_sql(db_table: str, con: Engine) -> datetime:
    """Get max processed date in sql table."""
    db_table = _safe_from_injection(db_table)
    dates = pd.read_sql_query(
        f"SELECT start_date FROM {db_table};", con, parse_dates=["period"]
    )
    max_date = dates.max()[0].to_pydatetime()
    return max_date


def human_readable_size(size: int, decimals: int = 1) -> str:
    """Transform size in bytes into human readable text."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1000:
            break
        size /= 1000
    return f"{size:.{decimals}f} {unit}"


class Era5:
    def __init__(self, variable: str, bounds: Tuple[float], cache_dir: str = None):
        """Copernicus climate data store client.

        Parameters
        ----------
        variable : str
            CDS variable name. See documentation for a list of available
            variables <https://confluence.ecmwf.int/display/CKB/ERA5-Land>.
        bounds : tuple of float
            Bounding box of interest as a tuple of float (lon_min, lat_min,
            lon_max, lat_max).
        cache_dir : str, optional
            Directory to cache downloaded files.
        """
        self.variable = variable
        self.bounds = bounds
        self.cache_dir = cache_dir
        self.cds_api_url = "https://cds.climate.copernicus.eu/api/v2"

    def init_cdsapi(self, api_key: str, api_uid: str):
        """Create a .cdsapirc in the HOME directory."""
        cdsapirc = os.path.join(os.getenv("HOME"), ".cdsapirc")
        with open(cdsapirc, "w") as f:
            f.write(f"url: {self.cds_api_url}\n")
            f.write(f"key: {api_uid}:{api_key}\n")
            f.write("verify: 0")
        logger.info(f"Created cdsapirc at {cdsapirc}")
        self.api = cdsapi.Client()

    def download(self, year: int, month: int, hours: List[str], dst_file: str) -> str:
        """Download product for a given date."""
        fname = os.path.basename(dst_file)

        if self.cache_dir:
            cachefs = filesystem(self.cache_dir)
            cachefp = os.path.join(self.cache_dir, fname)
            if cachefs.exists(cachefp):
                cachefs.get(cachefp, dst_file)
                logger.info(f"Loaded {fname} from cache")
                return dst_file

        request = {
            "format": "netcdf",
            "variable": self.variable,
            "year": year,
            "month": month,
            "day": [f"{d:02}" for d in range(1, 32)],
            "time": hours,
            "area": list(self.bounds),
        }
        self.api.retrieve("reanalysis-era5-land", request, dst_file)
        logger.info(f"Downloaded product into {dst_file}")

        # dataset should have data until last day of the month
        ds = xr.open_dataset(dst_file)
        n_days = monthrange(year, month)[1]
        if not ds.time.values.max() >= pd.to_datetime(datetime(year, month, n_days)):
            raise Era5MissingData(f"Missing data for period {year}{month}")

        if self.cache_dir:
            cachefs = filesystem(self.cache_dir)
            cachefp = os.path.join(self.cache_dir, fname)
            cachefs.put(dst_file, cachefp)
            logger.info(f"Cached {fname} into {self.cache_dir}")

        return dst_file


def zonal_stats(
    boundaries: gpd.GeoDataFrame,
    datafile: str,
    agg_function: Callable,
    column_id: str,
    column_name: str,
    variable_name: str = None,
):
    """Extract aggregated value for each area and time step.

    Parameters
    ----------
    boundaries : geodataframe
        Boundaries/areas for spatial aggregation.
    datafile : str
        Path to input dataset.
    agg_function : callable
        Function for spatial aggregation.
    column_id : str
        Column name in boundaries geodataframe with area ID.
    column_name : str
        Column name in boundaries geodataframe with area name.
    variable_name : str, optional
        Variable name in input dataset. If not set, first variable found will be
        used.

    Return
    ------
    dataframe
        Mean variable value as a dataframe of length (n_areas * n_days).
    """
    with rasterio.open(datafile) as src:
        transform = src.transform
        nodata = src.nodata
        height, width = src.height, src.width

    # build a raster index of areas, i.e. one binary raster mask per area
    # the index is a 3d array of shape (n_areas, height, width)
    areas = np.empty(shape=(len(boundaries), height, width), dtype=np.bool_)
    for i, geom in enumerate(boundaries.geometry):
        area = rasterize(
            [geom],
            out_shape=(height, width),
            fill=0,
            default_value=1,
            transform=transform,
            all_touched=True,
            dtype="uint8",
        )
        if np.count_nonzero(area) == 0:
            logger.warn(f"No cell covered by input geometry {i}")
        areas[i, :, :] = area == 1

    ds = xr.open_dataset(datafile)
    if not variable_name:
        variable_name = [var for var in ds.data_vars][0]

    records = []

    # get list of days in dataset from list of hours
    days = [pd.to_datetime(time).to_pydatetime().date() for time in ds.time.values]
    days = list(sorted(set(days)))

    for day in days:

        # we want all measurements for the current day
        day = pd.to_datetime(day)
        measurements = ds.sel(time=(ds.time >= day) & (ds.time < day + timedelta(1)))

        # for precipitation we want the accumulation during the day instead of the mean
        if variable_name == "tp":
            measurements = measurements.sum(dim="time")
        else:
            measurements = measurements.mean(dim="time")

        data = measurements[variable_name].values
        period = day.to_pydatetime().strftime("%Y-%m-%d")

        for i, area in enumerate(boundaries.index):

            # we can safely ignore negative values, temperature unit is K
            value = agg_function(
                data[(data >= 0) & (data != nodata) & (areas[i, :, :])]
            )

            records.append(
                {
                    "id": boundaries.at[area, column_id],
                    "name": boundaries.at[area, column_name],
                    "period": period,
                    "value": value,
                }
            )
            logger.debug(f"Aggregated measurements for area {i} (value = {value})")
        logger.info(f"Finished aggregation for period {period}")
    return pd.DataFrame(records)


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
            f"{n_features_orig - n_features} are invalid and were excluded from the analysis"
        )
    return geodf_


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


if __name__ == "__main__":
    cli()
