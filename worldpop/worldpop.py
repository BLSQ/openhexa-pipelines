import json
import logging
import os
import tempfile
import typing
from functools import cached_property

import click
import geopandas as gpd
import openhexa
import rasterio
import requests
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from rasterio.crs import CRS
from rasterstats import zonal_stats
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from s3fs import S3FileSystem
from urllib3.util import Retry

# common is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError as e:
    # ignore import error -> work anyway (but define logging)
    print(f"Common code import error: {e}")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)
oh = openhexa.OpenHexaContext()
dag = oh.get_current_dagrun()


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


def _s3_bucket_exists(fs: S3FileSystem, bucket: str) -> bool:
    """Check if a S3 bucket exists."""
    try:
        fs.info(f"s3://{bucket}")
        return True
    except FileNotFoundError:
        return False


@click.group()
def cli():
    pass


@cli.command()
@click.option("--country", "-c", type=str, required=True, help="Country ISO-A3 code")
@click.option(
    "--dataset", "-d", type=str, default="cic2020_100m", help="Worldpop dataset alias"
)
@click.option("--year", "-y", type=int, default=2020, help="Year of interest")
@click.option("--output-dir", "-o", type=str, required=True, help="Output directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files"
)
def download(country: str, dataset: str, year: int, output_dir: str, overwrite: bool):
    """Download WorldPop data."""
    fs = filesystem(output_dir)
    fs.makedirs(output_dir, exist_ok=True)

    if len(country) != 3:
        msg = f"{country} is not a valid ISO-A3 country code"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)

    if dataset not in DATASETS:
        msg = f"{dataset} is not a valid WorldPop dataset"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)

    fs.makedirs(output_dir, exist_ok=True)

    if len(fs.ls(output_dir)) > 0:
        if overwrite:
            msg = f"Data found in {output_dir} will be overwritten"
            logger.info(msg)
            dag.log_message("WARNING", msg)
        else:
            msg = f"Data found in {output_dir}, skipping download"
            logger.info(msg)
            dag.log_message("WARNING", msg)
            dag.progress_update(50)
            return

    worldpop = WorldPop(country, dataset)
    if year not in worldpop.years:
        msg = f"{year} not available for provided dataset"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_files = worldpop.download(tmp_dir, year)
        dag.progress_update(25)
        for tmp_file in tmp_files:
            fs.put(tmp_file, os.path.join(output_dir, os.path.basename(tmp_file)))
        dag.progress_update(50)

    dag.log_message("INFO", "Data download succeeded")


@cli.command()
@click.option(
    "--src-boundaries", type=str, required=True, help="Boundaries geographic file"
)
@click.option(
    "--src-population", type=str, required=True, help="Population raster file"
)
@click.option("--dst-file", type=str, required=True, help="Output file")
@click.option("--overwrite", is_flag=True, default=False)
def aggregate(src_boundaries: str, src_population: str, dst_file: str, overwrite: bool):
    """Spatial aggregation of population counts."""
    fs = filesystem(src_boundaries)

    # if src_population is a dir, just use the first found .tif
    if fs.isdir(src_boundaries):
        for fname in fs.ls(src_boundaries):
            if fname.lower().endswith(".tif") or fname.lower().endswith(".tiff"):
                src_boundaries = src_boundaries
                break

    with tempfile.TemporaryDirectory() as tmp_dir:

        fs = filesystem(src_boundaries)
        tmp_boundaries = os.path.join(tmp_dir, os.path.basename(src_boundaries))
        fs.get(src_boundaries, tmp_boundaries)

        fs = filesystem(src_population)
        tmp_population = os.path.join(tmp_dir, os.path.basename(src_population))
        fs.get(src_population, tmp_population)

        dag.progress_update(75)

        tmp_count = os.path.join(tmp_dir, "population_count.gpkg")
        count = count_population(
            src_boundaries=tmp_boundaries, src_population=tmp_population
        )
        count.to_file(tmp_count, driver="GPKG")
        fs = filesystem(dst_file)
        fs.put(tmp_count, dst_file)

        dag.log_message("INFO", "Spatial aggregagation succeeded")
        dag.progress_update(100)


API_URL = "https://hub.worldpop.org/rest"
DOWNLOAD_URL = "https://data.worldpop.org"

DATASETS = [
    "wpgp",  # Unconstrained individual countries 2000-2020 (100 m)
    "wpgpunadj",  # Unconstrained individual countries 2000-2020 UN adjusted (100 m)
    "wpic1km",  # Unconstrained individual countries 2000-2020 (1 km)
    "wpicuadj1km",  # Unconstrained individual countries 2000-2020 UN adjusted (1 km)
    "cic2020_100m",  # Constrained Individual countries 2020 (100 m)
    "cic2020_UNadj_100m",  # Constrained Individual countries 2020 UN adjusted (100 m)
]
DEFAULT_DATASET = "cic2020_100m"


class WorldPop:
    def __init__(self, country: str, dataset: str):

        if len(country) == 3:
            self.country = country
        else:
            msg = f"{country} is not a valid ISO-A3 country code"
            dag.log_message("ERROR", msg)
            raise ValueError(msg)

        if dataset in DATASETS:
            self.dataset = dataset
        else:
            msg = f"{dataset} is not a valid dataset"
            dag.log_message("ERROR", msg)
            raise ValueError(msg)

        retry_adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET"],
            )
        )
        self.s = requests.Session()
        self.s.mount("https://", retry_adapter)
        self.s.mount("http://", retry_adapter)

    @property
    def alias(self):
        """Dataset alias."""
        # constrained population maps
        if self.constrained:
            if self.resolution != 100:
                raise ValueError(
                    f"Constrained dataset only available for {self.resolution} m resolution"
                )
            if self.un_adjusted:
                return "cic2020_100m"
            else:
                return "cic2020_UNadj_100m"

        # unconstrained population maps
        else:
            if self.un_adjusted and self.resolution == 100:
                return "wpgpunadj"
            elif not self.un_adjusted and self.resolution == 100:
                return "wpgp"
            elif self.un_adjusted and self.resolution == 1000:
                return "wpic1km"
            elif not self.un_adjusted and self.resolution == 1000:
                return "wpicuadj1km"

    @cached_property
    def data(self) -> dict:
        """Available data."""
        r = self.s.get(f"{API_URL}/data/pop/{self.dataset}?iso3={self.country}")
        try:
            r.raise_for_status()
        except HTTPError as e:
            msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            dag.log_message("ERROR", msg)
            raise
        return r.json()["data"]

    @cached_property
    def years(self) -> typing.List[int]:
        """Available years."""
        years_ = []
        for ds in self.data:
            if "popyear" in ds:
                years_.append(int(ds["popyear"]))
        return years_

    def meta(self, year: int) -> dict:
        """Access dataset metadata for a given year."""
        if year not in self.years:
            msg = f"Year {year} not available for dataset {self.dataset}"
            dag.log_message("ERROR", msg)
            raise ValueError(msg)

        for ds in self.data:
            if ds.get("popyear") == str(year):
                return ds

    def _download(self, url: str, dst_file: str):
        """Download file from URL."""
        with tempfile.NamedTemporaryFile(suffix=dst_file.split(".")[-1]) as tmp_file:

            with self.s.get(url, stream=True, timeout=30) as r:

                try:
                    r.raise_for_status()
                except HTTPError:
                    dag.log_message("ERROR", "HTTP connection error")
                    raise
                with open(tmp_file.name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            fs = filesystem(dst_file)
            fs.put(tmp_file.name, dst_file)

    def download(
        self, dst_dir: str, year: int = 2020, overwrite: bool = False
    ) -> typing.List[str]:
        """Download dataset and metadata."""
        fs = filesystem(dst_dir)

        try:
            fs.makedirs(dst_dir, exist_ok=True)
        except PermissionError:
            dag.log_message(
                "ERROR", f"Permission error when trying to access {dst_dir}"
            )
            raise

        meta = self.meta(year)
        dst_files = []

        for url in meta["files"]:

            fname = url.split("/")[-1]
            dst_file = os.path.join(dst_dir, fname)

            if fs.exists(dst_file) and not overwrite:
                logger.warn(f"{dst_file} already exists, skipping download")
                dag.log_message(
                    "WARNING", f"{dst_file} already exists, skipping download"
                )
                continue

            logger.info(f"Downloading {url}")
            dag.log_message("INFO", f"Downloading {fname}")
            self._download(url=url, dst_file=dst_file)
            dst_files.append(dst_file)

        # write dataset metadata
        dst_file = os.path.join(dst_dir, "metadata.json")
        if not fs.exists(dst_file) or overwrite:
            with open(dst_file, "w") as f:
                json.dump(meta, f)
            dst_files.append(dst_file)

        return dst_files


def count_population(src_boundaries: str, src_population: str) -> gpd.GeoDataFrame:
    """Count population in each boundary.

    Parameters
    ----------
    src_boundaries : str
        Path to input boundaries file (geojson, geopackage, shapefile)
    src_population : str
        Path to input population raster (geotiff)

    Returns
    -------
    geodataframe
        Copy of the input boundaries geodataframe with population statistics.
    """
    try:
        boundaries = gpd.read_file(src_boundaries)
    except Exception:
        dag.log_message("ERROR", f"Cannot read {src_boundaries}")
        raise

    msg = f"Performing spatial aggregation on {len(boundaries)} polygons"
    logger.info(msg)
    dag.log_message("INFO", msg)

    with rasterio.open(src_population) as pop:

        # make sure boundaries and population raster have same CRS
        if not boundaries.crs:
            boundaries.crs = CRS.from_epsg(4326)
        if boundaries.crs != pop.crs:
            boundaries = boundaries.to_crs(pop.crs)

        stats = zonal_stats(
            boundaries.geometry,
            pop.read(1),
            affine=pop.transform,
            stats=["sum"],
            nodata=pop.nodata,
        )
        count = boundaries.copy()
        count["population"] = [s["sum"] for s in stats]

    return count


if __name__ == "__main__":

    cli()
