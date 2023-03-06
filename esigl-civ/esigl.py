import base64
import logging
import os
import re
from datetime import datetime, timedelta
from typing import List

import click
import openhexa
import pandas as pd
import requests
import requests_cache
from dateutil.relativedelta import relativedelta
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem
from sqlalchemy import create_engine

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


class Esigl:
    def __init__(self, url: str, username: str, password: str, cache: str = None):

        if cache:
            self.session = requests_cache.CachedSession(
                cache_name="http_cache",
                backend="sqlite",
                expire_after=timedelta(days=1),
                database=cache,
            )
        else:
            self.session = requests.Session()

        self.url = url.rstrip("/")
        self.authenticate(username, password)

    def authenticate(self, username: str, password: str):

        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.session.headers["Authorization"] = f"Basic {auth}"

    def get_monthly_report(self, year: int, month: int, program: str) -> pd.DataFrame:

        url = f"{self.url}/rest-api/stock-status/monthly"
        params = {"month": f"{month:02}", "year": str(year), "program": program}
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return pd.DataFrame.from_records(r.json()["report"])

    def get_metadata(self, metadata_type: str) -> pd.DataFrame:

        url = f"{self.url}/rest-api/lookup/{metadata_type}"
        r = self.session.get(url, params={"pageSize": 10000})
        r.raise_for_status()
        df = pd.DataFrame.from_records(r.json()[metadata_type])

        if "code" in df.columns:
            df.set_index("code", drop=True, inplace=True)
        elif "id" in df.columns:
            df.set_index("id", drop=True, inplace=True)

        return df

    def get_requisitions(self, facility_code: str) -> List[dict]:

        url = f"{self.url}/rest-api/requisitions"
        params = {"facilityCode": facility_code}
        r = self.session.get(url, params=params)
        r.raise_for_status()
        return r.json()["requisitions"]


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


def _check_iso_date(date: str) -> bool:
    """Check that date string is in ISO format."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date))


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--instance", type=str, required=True, help="ESIGL instance URL", envvar="ESIGL_URL"
)
@click.option(
    "--username",
    type=str,
    required=True,
    help="ESIGL username",
    envvar="ESIGL_USERNAME",
)
@click.option(
    "--password",
    type=str,
    required=True,
    help="ESIGL password",
    envvar="ESIGL_PASSWORD",
)
@click.option("--start", type=str, required=True, help="Start date in ISO format")
@click.option("--end", type=str, required=True, help="End date in ISO format")
@click.option("--program", type=str, multiple=True, help="Program short code")
@click.option("--output-dir", type=str, required=True, help="Output directory")
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files"
)
def download(
    instance: str,
    username: str,
    password: str,
    start: str,
    end: str,
    program: List[str],
    output_dir: str,
    overwrite: bool,
):
    """Download monthly reports."""
    output_dir = output_dir.lstrip("/")

    # check provided start and end dates
    if not _check_iso_date(start):
        msg = "Start date must be in ISO format"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)
    if not _check_iso_date(end):
        msg = "End date must be in ISO format"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)

    if not program:
        msg = "No program provided"
        dag.log_message("ERROR", "No program provided")
        raise ValueError(msg)

    fs = filesystem(output_dir)

    # check provided output dir
    try:
        fs.exists(output_dir)
    except PermissionError:
        dag.log_message(
            "ERROR", "Permission error when trying to access output directory"
        )
        raise

    # remove data from previous run if it is detected in the output dir
    # if overwrite is set to False, raise an error
    raw_data_dir = f"{output_dir}/raw"
    if fs.exists(raw_data_dir):
        if overwrite:
            for f in fs.glob(f"{raw_data_dir}/*.csv"):
                fs.rm(f)
            for f in fs.glob(f"{raw_data_dir}/metadata/*.csv"):
                fs.rm(f)
        else:
            return FileExistsError("Output directory is not empty")

    # parse dates
    start = datetime.strptime(start[:7], "%Y-%m")
    end = datetime.strptime(end[:7], "%Y-%m")
    if end < start:
        msg = "End date is inferior to start date"
        dag.log_message("ERROR", msg)
        raise ValueError(msg)

    esigl = Esigl(url=instance, username=username, password=password)

    # save metadata
    metadata_types = [
        "programs",
        "facilities",
        "geographic-zones",
        "processing-periods",
        "products",
        "product-categories",
        "program-products",
    ]
    dag.log_message("INFO", "Downloading ESIGL metadata")
    for mtype in metadata_types:
        df = esigl.get_metadata(mtype)
        fpath = os.path.join(output_dir, "metadata", f"{mtype}.csv")
        with fs.open(fpath, "w") as f:
            df.to_csv(f, index=True)

    # build range of dates for which we want to download reports
    date_range = []
    date = start
    while date < end + relativedelta(months=1):
        date_range.append(date)
        date += relativedelta(months=1)

    i = 0
    for date in date_range:
        dag.log_message(
            "INFO", f"Downloading monthly stock status for {date.strftime('%Y-%m')}"
        )

        for prg in program:

            report = esigl.get_monthly_report(
                year=date.year, month=date.month, program=prg
            )

            fpath = os.path.join(
                output_dir, f"stock_status_{date.strftime('%Y%m')}_{prg}.csv"
            )

            if not report.empty:
                with fs.open(fpath, "w") as f:
                    report.to_csv(f, index=False)

            # progress from 0 to 50% at the end of the loop
            i += 1
            progress = i / (len(date_range) * len(program)) * 50

            # openhexa-sdk return an error if the progress is < 1
            # todo: fix this in openhexa-sdk instead
            if progress < 1:
                progress = 1

            dag.progress_update(round(progress))

    dag.log_message("INFO", "Successful data download")


def _from_unix_timestamp(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp / 1000)


def _get_parent_name(
    zone_name: str, geographic_zones: pd.DataFrame, level: int = 3
) -> str:
    """Get name of geographic parent."""
    df = geographic_zones[
        (geographic_zones["name"] == zone_name) & (geographic_zones["levelId"] == level)
    ]
    if len(df) != 1:
        return None

    parent_id = int(df.parentId.values[0])
    df = geographic_zones[geographic_zones["id"] == parent_id]
    if len(df) != 1:
        return None

    return df.name.values[0]


def _get_facility_code(facility_name: str, facilities: pd.DataFrame) -> str:
    """Get facility code from facility name."""
    df = facilities[facilities["name"] == facility_name]
    if len(df):
        return df.index.values[0]
    else:
        return None


def _get_product_category(
    product_id: str, product_categories: pd.DataFrame, program_products: pd.DataFrame
) -> str:
    """Get product category name from product id."""
    df = program_products[program_products["product"] == product_id]
    if df.empty:
        return None
    product_category_id = df["productCategoryId"].values[0]

    df = product_categories[product_categories.id == product_category_id]
    if len(df) != 1:
        return None

    return df["name"].values[0]


def _stock_status(msd: float, sdu: float, cmm: float) -> str:
    if sdu > 0 and cmm == 0:
        return "Stock dormant"
    elif msd == 0:
        return "Rupture"
    elif msd > 0 and msd <= 1:
        return "En bas du PCU"
    elif msd > 1 and msd < 2:
        return "Entre PCU et MIN"
    elif msd >= 2 and msd <= 4:
        return "Bien stocké"
    elif msd > 4:
        return "Surstock"


def _is_tracer(product_code: str, products: pd.DataFrame) -> bool:
    return products.loc[product_code]["tracer"]


def _get_dosage_unit(product_code: str, products: pd.DataFrame) -> str:
    return products.loc[product_code]["dispensingUnit"]


def process_report(
    src_report: pd.DataFrame,
    facilities: pd.DataFrame,
    geographic_zones: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Rename columns and transform values as needed."""
    df = src_report.copy()
    COLUMNS = {
        "programCode": "code_programme",
        "programName": "programme",
        "reportPeriodName": "periode",
        "reportedDate": "date",
        "facilityName": "code_ets",
        "district": "district",
        "productCode": "code_produit",
        "product": "produit",
        "beginningBalance": "stock_initial",
        "quantityReceived": "quantite_recue",
        "quantityDispensed": "quantite_utilisee",
        "stockOutDays": "jours_de_rupture",
        "stockInHand": "sdu",
        "amc": "cmm_esigl",
        "quantityRequested": "quantite_commandee",
        "quantityApproved": "quantite_approuvee",
        "mos": "msd",
    }
    df = df.rename(columns=COLUMNS)
    df = df.drop(columns=[col for col in df.columns if col not in COLUMNS.values()])
    df["code_ets"] = df["code_ets"].apply(lambda x: _get_facility_code(x, facilities))
    df["date"] = df["date"].apply(_from_unix_timestamp)
    df["region"] = df["district"].apply(lambda x: _get_parent_name(x, geographic_zones))
    df["unite_de_rapportage"] = df["code_produit"].apply(
        lambda x: _get_dosage_unit(x, products)
    )
    df["etat_du_stock"] = df.apply(
        lambda row: _stock_status(row.msd, row.cmm_esigl, row.sdu), axis=1
    )
    return df


@cli.command()
@click.option("--input-dir", type=str, required=True, help="Input directory")
@click.option("--output-dir", type=str, help="Output directory")
@click.option("--output-db", type=str, help="Output database")
@click.option("--output-db-table", type=str, help="Output database table")
def transform(input_dir: str, output_dir: str, output_db: str, output_db_table: str):
    """Merge and transform monthly stock status reports."""
    # load required metadata
    fs = filesystem(input_dir)
    with fs.open(os.path.join(input_dir, "metadata", "facilities.csv")) as f:
        facilities = pd.read_csv(f, index_col=0)
    with fs.open(os.path.join(input_dir, "metadata", "geographic-zones.csv")) as f:
        geographic_zones = pd.read_csv(f, index_col=0)
    with fs.open(os.path.join(input_dir, "metadata", "products.csv")) as f:
        products = pd.read_csv(f, index_col=0)

    # merge monthly stock status reports
    src_files = fs.glob(os.path.join(input_dir, "stock_status_*.csv"))
    dag.log_message("INFO", "Merging stock status reports")
    merge = pd.DataFrame()
    for src in src_files:
        with fs.open(src) as f:
            df = pd.read_csv(f)
            if df.empty:
                continue
            if merge.empty:
                merge = df
            else:
                merge = pd.concat([merge, df])
    dag.progress_update(75)

    dag.log_message("INFO", "Processing final report")
    merge = process_report(
        src_report=merge,
        facilities=facilities,
        geographic_zones=geographic_zones,
        products=products,
    )
    dag.progress_update(100)

    if output_dir:
        fpath = os.path.join(output_dir, "stock_status.csv")
        dag.log_message("INFO", f"Writing final report into {fpath}")
        fs = filesystem(fpath)
        with fs.open(fpath, "w") as f:
            merge.to_csv(f, index=False)
        dag.add_outputfile(os.path.basename(fpath), fpath)

    if output_db and output_db_table:
        dag.log_message("INFO", f"Writing final report into {output_db}")
        db_slug = output_db.upper().replace("-", "_")
        db_name = os.getenv(f"POSTGRESQL_{db_slug}_DATABASE")
        db_host = os.getenv(f"POSTGRESQL_{db_slug}_HOSTNAME")
        db_port = os.getenv(f"POSTGRESQL_{db_slug}_PORT")
        db_user = os.getenv(f"POSTGRESQL_{db_slug}_USERNAME")
        db_pass = os.getenv(f"POSTGRESQL_{db_slug}_PASSWORD")
        con = create_engine(
            f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        )
        merge.to_sql(output_db_table, con, if_exists="replace")

    dag.log_message("INFO", "Successful data processing")


if __name__ == "__main__":
    cli()
