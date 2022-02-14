"""Download and extract data from CPC Global Daily Temperarature.

See <https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html>.
"""


import datetime
import enum
import logging
import os
from datetime import date
from typing import List

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray
from pyproj import CRS
from rasterstats import zonal_stats

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


@click.group()
def cli():
    pass


@cli.command()
@click.option("--start", type=str, required=True, help="start date of period to sync")
@click.option("--end", type=str, required=True, help="end date of period to sync")
@click.option("--output-dir", type=str, required=True, help="output data directory")
@click.option(
    "--variable", type=str, required=False, default="tmax", help="temperature variable"
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def sync(start: str, end: str, output_dir: str, variable: str, overwrite: bool):
    """Sync yearly temperature data sets."""
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    os.makedirs(output_dir, exist_ok=True)
    gdt = GlobalDailyTemperature(data_dir=output_dir)
    gdt.sync(
        start_year=start.year, end_year=end.year, variable=variable, overwrite=overwrite
    )


@cli.command()
@click.option("--start", type=str, required=True, help="start date of analysis")
@click.option("--end", type=str, required=True, help="end date of analysis")
@click.option("--areas", type=str, required=True, help="aggregation areas")
@click.option("--input-dir", "-i", type=str, required=True, help="input data directory")
@click.option(
    "--output-file", "-o", type=str, required=True, help="output data directory"
)
@click.option(
    "--areas-index",
    type=str,
    required=False,
    default="ou_uid",
    help="column with unique index",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def daily(
    start: str,
    end: str,
    areas: str,
    input_dir: str,
    output_file: str,
    areas_index: str,
    overwrite: bool,
):
    """Daily spatial extract of temperature data."""
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")

    # if output CSV already exists, start analysis from latest date
    # available and append data to existing file
    mode = "w"
    header = True
    if os.path.isfile(output_file) and not overwrite:
        daily = pd.read_csv(output_file, index_col=0)
        daily.index = pd.to_datetime(daily.index)
        start = max(start, daily.index.max().date())
        mode = "a"
        header = False

    areas = gpd.read_file(areas)
    if not areas.crs:
        areas.crs = CRS.from_epsg(4326)
    # index of areas geodataframe is used as a label in the output dataframe
    areas.set_index(areas_index, inplace=True)

    gdt = GlobalDailyTemperature(data_dir=input_dir)
    mean = gdt.calc_daily_stats(
        areas=areas, start=start, end=end, variable="tmax", statistic="mean"
    )
    mean.to_csv(output_file, mode=mode, header=header)


@cli.command()
@click.option("--start", type=str, required=True, help="start date of extraction")
@click.option("--end", type=str, required=True, help="end date of extraction")
@click.option("--areas", type=str, required=True, help="aggregation areas")
@click.option("--daily", type=str, required=True, help="input daily data file")
@click.option("--areas-index", type=str, required=False, help="index column name")
@click.option(
    "--overwrite", is_flag=True, default=False, help="overwrite existing files"
)
def weekly(
    start: str, end: str, areas: str, daily: str, areas_index: str, overwrite: bool
):
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    daily = pd.read_csv(daily, index_col=0)
    daily.index = pd.to_datetime(daily.index)
    areas = gpd.read_file(areas)
    areas.set_index(areas_index, inplace=True)

    for epiweek in epiweek_range(start, end):
        weekly = daily[epiweek.start : epiweek.end].mean()  # noqa
        weekly.rename(f"{epiweek.year}W{epiweek.week:0>2}", inplace=True)
        areas = areas.merge(weekly, left_index=True, right_index=True)


@cli.command()
@click.option("--start", type=str, required=True, help="start date of extraction")
@click.option("--end", type=str, required=True, help="end date of extraction")
@click.option("--areas", type=str, required=True, help="aggregation areas")
@click.option("--daily", type=str, required=True, help="input daily data file")
@click.option("--areas_index", type=str, required=False, help="index column name")
def monthly(
    start: str, end: str, areas: str, daily: str, areas_index: str, overwrite: bool
):
    pass


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
            yield self.start + datetime.timedelta(days=wday)

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

        week_day = (
            date_object - datetime.timedelta(days=systems[system.upper()])
        ).isoweekday()
        week_day = 0 if week_day == 7 else week_day  # Week : Sun = 0 ; Sat = 6

        # Start the weekday on Sunday (CDC)
        self.start = date_object - datetime.timedelta(days=week_day)
        # End the weekday on Saturday (CDC)
        self.end = self.start + datetime.timedelta(days=6)

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
        day += datetime.timedelta(days=1)
    return epiweeks


class ProductNotAvailable(Exception):
    pass


class GlobalDailyTemperature:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.base_url = "https://downloads.psl.noaa.gov/Datasets/cpc_global_temp"

    def get_download_url(self, year: int, variable: str = "tmax") -> str:
        """Get download URL for the CPC product (one file per year).

        Parameters
        ----------
        year : int
            Year of interest (from 1979).
        variable : str, optional
            Either tmax (maximum daily temperature) or tmin (minimum temperature
            of the day).

        Return
        ------
        str
            Download URL.
        """
        if variable not in ["tmax", "tmin"]:
            raise ValueError(f"`{variable}` is not a valid CPC Global Temp variable.")
        if year < 1979 or year > datetime.datetime.now().year:
            raise ValueError(f"Year {year} not supported.")
        return f"{self.base_url}/{variable}.{year}.nc"

    def download(
        self, dst_file: str, year: int, variable: str = "tmax", overwrite: bool = False
    ) -> str:
        """Download yearly CPC global temperature product.

        Parameters
        ----------
        dst_file : str
            Path to output file.
        year : int
            Year of interest (from 1979).
        variable : str, optional
            Temperature variable (tmin or tmax, default="tmax").
        overwrite : bool, optional
            Overwrite existing files (default=False). If set to False,
            remote file is only downloaded if local and remote file
            sizes differ.

        Return
        ------
        str
            Path to downloaded dataset.
        """
        url = self.get_download_url(year, variable)
        logger.info(f"Product URL: {url}.")
        r = requests.head(url)
        if r.status_code == 404:
            raise ProductNotAvailable(f"Product not available for year {year}.")
        r.raise_for_status()
        content_length = int(r.headers.get("Content-Length"))

        if os.path.isfile(dst_file) and not overwrite:
            if os.path.getsize(dst_file) == content_length:
                logger.info(f"Dataset {dst_file} is already up-to-date.")
                return dst_file

        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dst_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        logger.info(f"Downloaded {dst_file} from {url}.")

        if os.path.getsize(dst_file) != content_length:
            raise IOError(f"Remote and local sizes of {dst_file} differ.")

        return dst_file

    def sync(
        self,
        start_year: int,
        end_year: int,
        variable: str = "tmax",
        overwrite: bool = False,
    ):
        """Download yearly CPC global temperature products.

        Parameters
        ----------
        start_year : int
            Period start year.
        end_year : int
            Period end year.
        variable : str, optional
            Temperature variable (tmax or tmin, default="tmax").
        overwrite: bool, optional
            Overwrite existing files (default=False).
        """
        if variable not in ["tmax", "tmin"]:
            raise ValueError(f"`{variable}` is not a valid CPC Global Temp variable.")
        for year in (start_year, end_year):
            if year < 1979 or year > datetime.datetime.now().year:
                raise ValueError(f"Year {year} not supported.")

        for year in range(start_year, end_year + 1):

            try:
                self.download(
                    dst_file=os.path.join(self.data_dir, f"{variable}.{year}.nc"),
                    year=year,
                    variable=variable,
                    overwrite=overwrite,
                )
            except ProductNotAvailable:
                logger.warn(f"Product not available for year {year}.")

    def calc_daily_stats(
        self,
        areas: gpd.GeoDataFrame,
        start: datetime.date,
        end: datetime.date,
        variable: str = "tmax",
        statistic: str = "mean",
    ) -> pd.DataFrame:
        """Compute zonal stats for input geometries.

        Compute min, max, mean and count spatial aggregation
        statistics.

        Parameters
        ----------
        areas : geodataframe
            Spatial aggregation areas.
        start : date
            Starting date for temporal aggregation.
        end : date
            Ending date for temporal aggregation.
        variable : str, optional
            Temperature variable (tmin or tmax, default="tmax").
        statistic : str, optional
            Statistic for spatial aggregation (default="mean").
            Also available: "max", "min" or "count".

        Return
        ------
        dataframe
           dataframe of shape (n_days, n_areas).
        """
        timeserie = pd.DataFrame(columns=areas.index)

        # get some common raster metadata
        fp = os.path.join(self.data_dir, f"{variable}.{start.year}.nc")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Data not found at {fp}.")
        with rasterio.open(fp) as src:
            transform = src.transform
            nodata = src.nodata
        crs = CRS.from_string(
            "+proj=longlat +ellps=WGS84 +pm=-360 +datum=WGS84 +no_defs"
        )

        geoms = [geom.__geo_interface__ for geom in areas.to_crs(crs).geometry]

        day = start
        while day <= end:

            fp = os.path.join(self.data_dir, f"{variable}.{day.year}.nc")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Data not found at {fp}.")

            ds = xarray.open_dataset(fp)
            data = ds[variable].sel(time=np.datetime64(day)).values

            stats = zonal_stats(
                vectors=geoms,
                raster=data,
                affine=transform,
                all_touched=True,
                nodata=nodata,
                stats=[statistic],
            )

            daily_data = pd.DataFrame(
                index=[day],
                columns=areas.index,
                data=[[area[statistic] for area in stats]],
            )

            timeserie = pd.concat((timeserie, daily_data), axis=0)
            day += datetime.timedelta(days=1)

        timeserie.index = pd.to_datetime(timeserie.index)
        return timeserie


if __name__ == "__main__":
    cli()
