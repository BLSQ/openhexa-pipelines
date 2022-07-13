import logging
import os
import string
import tempfile
from datetime import datetime
from typing import Callable, List, Sequence

import click
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from appdirs import user_cache_dir
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from rasterio.features import rasterize
from rasterio.merge import merge
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

APP_NAME = "openhexa-pipelines-ndvi"
APP_AUTHOR = "bluesquare"

BASE_URL = "https://e4ftl01.cr.usgs.gov/MOLT/"
EARTHDATA_URL = "https://urs.earthdata.nasa.gov"


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
@click.option("--csv", type=str, required=False, help="output CSV file")
@click.option("--db-user", type=str, required=False, help="database username")
@click.option("--db-password", type=str, required=False, help="database password")
@click.option("--db-host", type=str, required=False, help="database hostname")
@click.option("--db-port", type=int, required=False, help="database port")
@click.option("--db-name", type=str, required=False, help="database name")
@click.option("--db_table", type=str, required=False, help="database table")
@click.option(
    "--earthdata-username",
    type=str,
    required=True,
    envvar="EARTHDATA_USERNAME",
    help="nasa earthdata username",
)
@click.option(
    "--earthdata-password",
    type=str,
    required=True,
    envvar="EARTHDATA_PASSWORD",
    help="nasa earthdata password",
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
    csv: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_table: str,
    earthdata_username: str,
    earthdata_password: str,
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

    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")

    # find latest processed date in csv and/or postgres table
    # only later dates will be processed except if overwrite = False
    fs = filesystem(csv)
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
        csv_mode = "w"  # write mode
        if fs.exists(csv):
            fs.get(csv, tmp_file.name)
            max_date = _max_date_csv(tmp_file.name)
            if max_date > start:
                start = max_date + relativedelta(months=1)
                csv_mode = "a"  # append mode

    lpdaac = LPDAAC(product="MOD13A3.061", cache_dir=cache_dir, timeout=60)
    lpdaac.login(earthdata_username, earthdata_password)

    boundaries_data.set_index(column_id, inplace=True, drop=False)
    tileindex = gpd.read_file(
        os.path.join(os.path.dirname(__file__), "tile_index.gpkg")
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        data = []
        date = start
        while date <= end:

            files = []
            for tile in lpdaac.find(tiles=tileindex, areas=boundaries_data):
                url = lpdaac.get_url(tile, date.year, date.month)
                fname = url.split("/")[-1]
                dst_file = lpdaac.download(url, os.path.join(tmp_dir, fname))
                files.append(dst_file)

            mosaic = merge_tiles(
                files, dst_file=os.path.join(tmp_dir, f"{date.strftime('%Y%m%d')}.tif")
            )

            stats = zonal_aggregation(boundaries_data, mosaic, np.mean)
            for area in boundaries_data.index:
                data.append(
                    {
                        "id": area,
                        "name": boundaries_data.at[area, column_name],
                        "period": date.strftime("%Y%m"),
                        "start_date": (date).strftime("%Y-%m-%d"),
                        "end_date": (
                            start + relativedelta(months=1) - relativedelta(days=1)
                        ).strftime("%Y-%m-%d"),
                        "ndvi": round(stats[area], 1),
                    }
                )

            date = date + relativedelta(months=1)

        df = pd.DataFrame(data)
        if csv:
            tmp_file = os.path.join(tmp_dir, "extract.csv")
            df.to_csv(tmp_file, index=False, mode=csv_mode)
            fs.put(tmp_file, csv)
        if db_table:
            db_table_safe = _safe_from_injection(db_table)
            con = create_engine(
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            df.to_sql(db_table_safe, con, index=False, if_exists="append")


class LPDAACError(Exception):
    pass


def _safe_from_injection(db_table: str) -> str:
    """Remove potential SQL injection."""
    return "".join(
        [c for c in db_table if c in string.ascii_letters + string.digits + "_"]
    )


def _max_date_csv(src_file: str) -> datetime:
    """Get max processed date in csv file."""
    data = pd.read_csv(src_file)
    max_date = pd.to_datetime(data.start_date).max().to_pydatetime()
    return max_date


def _max_date_sql(db_table: str, con: Engine) -> datetime:
    """Get max processed date in sql table."""
    db_table = _safe_from_injection(db_table)
    dates = pd.read_sql_query(
        f"SELECT start_date FROM {db_table};", con, parse_dates=["start_date"]
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


class LPDAAC:
    def __init__(
        self, product: str = "MOD13A3.061", cache_dir: str = None, timeout: int = 60
    ):
        self.product = product
        self.cache = {}  # temporary cache for html indexes
        self.cache_dir = cache_dir  # permanent storage cache
        self.session = requests.Session()
        self.timeout = timeout

    @property
    def cache_fs(self):
        if self.cache_dir.startswith("s3://"):
            return fsspec.filesystem("s3")
        elif self.cache_dir.startswith("gcs://"):
            return fsspec.filesystem("gcs")
        else:
            return fsspec.filesystem("file")

    def login(self, username: str, password: str):
        """Login to EarthData."""
        # extract authenticity token from login page
        with self.session.get(EARTHDATA_URL, timeout=self.timeout) as r:
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            token = ""
            for element in soup.find_all("input"):
                if element.attrs.get("name") == "authenticity_token":
                    token = element.attrs.get("value")
                    break
            if not token:
                raise Exception("Token not found in EarthData login page.")

        r = self.session.post(
            EARTHDATA_URL + "/login",
            timeout=self.timeout,
            data={
                "username": username,
                "password": password,
                "authenticity_token": token,
            },
        )
        r.raise_for_status()
        logger.info(f"EarthData: logged in as {username}")

    def find(self, tiles: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> List[str]:
        """Find tiles required to cover the areas of interest."""
        if not tiles.crs:
            tiles = tiles.set_crs("EPSG:4326")
        if not areas.crs:
            areas = areas.set_crs("EPSG:4326")

        if areas.crs != tiles.crs:
            areas = areas.to_crs(tiles.crs)
            logger.info(f"Reprojected areas to CRS {tiles.crs}")

        result = tiles[tiles.intersects(areas.unary_union)]
        identifiers = [f"h{h:02}v{v:02}" for h, v in zip(result.ih, result.iv)]
        logger.info(f"Found {len(identifiers)} tiles")

        if not len(identifiers):
            raise LPDAACError("No MODIS tile found for the provided areas.")

        return identifiers

    def get_url(self, tile: str, year: int, month: int) -> List[str]:
        """Get download URL of a tile."""
        # there is 1 html index page available for each date
        # as the URL cannot be guessed, it must be found inside the index based
        # on the tile identifier
        dir_url = f"{BASE_URL}/{self.product}/{year}.{month:02}.01/"
        logger.debug(f"Directory URL: {dir_url}")

        # check if the index has already been parsed
        cache_id = f"{year}.{month:02}.01"
        if cache_id in self.cache:
            soup = BeautifulSoup(self.cache[cache_id], features="html.parser")
        else:
            with requests.get(dir_url) as r:
                r.raise_for_status()
                body = r.text
                self.cache[cache_id] = body
                soup = BeautifulSoup(body, features="html.parser")

        # find the URL
        url = None
        for a in soup.find_all("a"):
            href = a.attrs.get("href", "")
            if tile in href and href.endswith(".hdf"):
                url = dir_url + href

        return url

    def download(self, url: str, dst_file: str):
        """Download a tile."""
        fname = url.split("/")[-1]

        # try to get file from cache
        if self.cache_dir:
            fpath = os.path.join(self.cache_dir, fname)
            if self.cache_fs.exists(fpath):
                self.cache_fs.get(fpath, dst_file)
                logger.info(f"Copied {fname} from cache")
                return dst_file

        if not os.path.exists(os.path.dirname(dst_file)):
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with self.session.get(url, timeout=self.timeout) as r:
            r.raise_for_status()
            with open(dst_file, "wb") as f:
                f.write(r.content)
        logger.info(
            f"Downloaded {url} into {dst_file} ({human_readable_size(os.path.getsize(dst_file))})"
        )

        if self.cache_dir:
            fpath = os.path.join(self.cache_dir, fname)
            self.cache_fs.makedirs(self.cache_dir, exist_ok=True)
            self.cache_fs.put(dst_file, fpath)
            logger.info(f"Cached {fname} into {self.cache_dir}")

        return dst_file


def merge_tiles(tiles: Sequence[str], dst_file: str):
    """Merge input tiles into a single geotiff mosaic.

    Parameters
    ----------
    tiles : list of str
        Paths to input tiles.
    dst_file : str
        Path to output geotiff.

    Return
    ------
    str
        Path to output geotiff.
    """
    # input data is located inside a specific subdataset of the HDF files
    DRIVER = "HDF4_EOS:EOS_GRID"
    SUBDATASET = 'MOD_Grid_monthly_1km_VI:"1 km monthly NDVI"'
    datasets = [rasterio.open(f'{DRIVER}:"{tile}":{SUBDATASET}') for tile in tiles]

    # initialize output raster metadata
    meta = datasets[0].meta
    meta["driver"] = "GTiff"
    meta["compress"] = "zstd"

    if not meta.get("nodata"):
        logger.warn("Undefined nodata value")

    array, affine = merge(datasets)
    logger.info(f"Merged {len(tiles)} into a mosaic of shape {array.shape}")
    meta.update(height=array.shape[1], width=array.shape[2], transform=affine)
    with rasterio.open(dst_file, "w", **meta) as dst:
        dst.write(array)

    return dst_file


def zonal_aggregation(
    areas: gpd.GeoDataFrame, src_raster: str, agg_function: Callable
) -> pd.Series:
    """Compute mean measurement for each input area.

    Parameters
    ----------
    areas : geodataframe
        Input geodataframe with geometries.
    src_raster : str
        Path to input raster.
    agg_function : callable
        Aggregation function.

    Return
    ------
    series
        Output statistics as a series of length (n_areas).
    """
    with rasterio.open(src_raster) as src:
        data = src.read(1)
        height, width = data.shape
        transform = src.transform
        nodata = src.nodata

    zones = np.empty(shape=(len(areas), height, width), dtype=np.bool_)
    logger.info(f"Rasterizing masks for {len(zones)} geometries")

    stats = []
    for i, geom in areas.geometry.items():
        zone = rasterize(
            [geom],
            out_shape=(height, width),
            fill=0,
            default_value=1,
            transform=transform,
            all_touched=True,
            dtype="uint8",
        )
        if np.count_nonzero(zone) == 0:
            logger.warn(f"No cell covered by input geometry (index={i}).")
        measurements = data[(data != nodata) & (zone == 1)].ravel()
        stats.append(agg_function(measurements))
        logger.info(
            f"Aggregated {np.count_nonzero(measurements)} measurements for area {i}"
        )

    return pd.Series(index=areas.index, data=stats)


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


if __name__ == "__main__":
    cli()
