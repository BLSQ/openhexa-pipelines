import json
import logging
import os
import re
import tempfile
import typing
from itertools import product

import click
import geopandas as gpd
import pandas as pd
from api import Api
from click.types import Choice
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from period import Period, get_range
from s3fs import S3FileSystem
from shapely.geometry import shape

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


class DHIS2ExtractError(Exception):
    """DHIS2 extraction error."""


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
@click.option(
    "--instance",
    "-i",
    type=str,
    required=True,
    help="DHIS2 instance URL.",
    envvar="DHIS2_INPUT_URL",
)
@click.option(
    "--username",
    "-u",
    type=str,
    required=True,
    envvar="DHIS2_INPUT_USERNAME",
    help="DHIS2 username.",
)
@click.option(
    "--password",
    "-p",
    type=str,
    required=True,
    envvar="DHIS2_INPUT_PASSWORD",
    help="DHIS2 password.",
)
@click.option("--output-dir", "-o", type=str, required=True, help="Output directory.")
@click.option("--start", "-s", type=str, help="Start date in ISO format.")
@click.option("--end", "-e", type=str, help="End date in ISO format.")
@click.option("--period", "-pe", type=str, multiple=True, help="DHIS2 period.")
@click.option(
    "--org-unit", "-ou", type=str, multiple=True, help="Organisation unit UID."
)
@click.option(
    "--org-unit-group",
    "-oug",
    type=str,
    multiple=True,
    help="Organisation unit group UID.",
)
@click.option(
    "--org-unit-level", "-lvl", type=int, multiple=True, help="Organisation unit level."
)
@click.option("--dataset", "-ds", type=str, multiple=True, help="Dataset UID.")
@click.option(
    "--data-element", "-de", type=str, multiple=True, help="Data element UID."
)
@click.option(
    "--data-element-group",
    "-deg",
    type=str,
    multiple=True,
    help="Data element group UID.",
)
@click.option("--indicator", "-in", type=str, multiple=True, help="Indicator UID.")
@click.option(
    "--indicator-group", "-ing", type=str, multiple=True, help="Indicator group UID."
)
@click.option(
    "--attribute-option-combo",
    "-aoc",
    type=str,
    multiple=True,
    help="Attribute option combo UID.",
)
@click.option(
    "--category-option-combo",
    "-coc",
    type=str,
    multiple=True,
    help="Category option combo UID.",
)
@click.option("--program", "-prg", type=str, multiple=True, help="Program UID.")
@click.option("--from-json", type=str, help="Load parameters from a JSON file.")
@click.option(
    "--children/--no-children",
    is_flag=True,
    default=False,
    help="Include children of selected org units.",
)
@click.option(
    "--mode",
    "-m",
    type=Choice(["analytics", "analytics-raw", "raw"], case_sensitive=False),
    default="analytics",
    help="DHIS2 API endpoint to use",
)
@click.option(
    "--metadata-only", is_flag=True, default=False, help="Only download metadata."
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing file."
)
def download(
    instance: str,
    username: str,
    password: str,
    output_dir: str,
    start: str,
    end: str,
    period: typing.Sequence[str],
    org_unit: typing.Sequence[str],
    org_unit_group: typing.Sequence[str],
    org_unit_level: typing.Sequence[str],
    dataset: typing.Sequence[str],
    data_element: typing.Sequence[str],
    data_element_group: typing.Sequence[str],
    indicator: typing.Sequence[str],
    indicator_group: typing.Sequence[str],
    attribute_option_combo: typing.Sequence[str],
    category_option_combo: typing.Sequence[str],
    program: typing.Sequence[str],
    from_json: str,
    children: bool,
    mode: str,
    metadata_only: bool,
    overwrite: bool,
):
    """Download data from a DHIS2 instance via its web API."""
    dhis = DHIS2(instance, username, password, timeout=120)
    output_dir = output_dir.rstrip("/")
    fs = filesystem(output_dir)

    # Check for existing files at the beginning of the function to
    # avoid making useless API calls.
    for fname in (
        "data_value_sets.csv",
        "analytics.csv",
        "analytics_raw_data.csv",
        "metadata.json",
    ):
        fpath = f"{output_dir}/{fname}"
        if fs.exists(fpath) and not overwrite:
            raise FileExistsError(f"File {fpath} already exists.")

    # S3FileSystem.mkdirs() will automatically create a new
    # bucket if permissions allow it and we do not want that.
    if output_dir.startswith("s3://"):
        bucket = output_dir.split("//")[-1].split("/")[0]
        if not _s3_bucket_exists(fs, bucket):
            raise DHIS2ExtractError(f"S3 bucket {bucket} does not exist.")

    fs.mkdirs(output_dir, exist_ok=True)

    # Load dimension parameters from JSON file.
    # If key is present in the JSON file, it overrides the
    # CLI parameters.
    if from_json:
        fs = filesystem(from_json)
        with fs.open(from_json) as f:
            params = json.load(f)
        period = params.get("period", period)
        org_unit = params.get("org-unit", org_unit)
        org_unit_group = params.get("org-unit-group", org_unit_group)
        org_unit_level = params.get("org-unit-level", org_unit_level)
        dataset = params.get("dataset", dataset)
        data_element = params.get("data-element", data_element)
        data_element_group = params.get("data-element-group", data_element_group)
        indicator = params.get("indicator", indicator)
        indicator_group = params.get("indicator-group", indicator_group)
        attribute_option_combo = params.get(
            "attribute-option-combo", attribute_option_combo
        )
        category_option_combo = params.get(
            "category-option-combo", category_option_combo
        )
        program = params.get("program", program)

    output_meta = f"{output_dir}/metadata.json"
    with fs.open(output_meta, "w") as f:
        logger.debug(f"Writing metadata to {output_meta}.")
        json.dump(dhis.metadata, f)

    if metadata_only:
        return

    # The dataValueSets endpoint does not support data elements UIDs as parameters,
    # only datasets.
    if mode.lower() == "raw":

        csv = dhis.data_value_sets(
            datasets=dataset,
            periods=period,
            start_date=start,
            end_date=end,
            org_units=org_unit,
            org_unit_groups=org_unit_group,
            org_unit_levels=org_unit_level,
            data_elements=data_element,
            data_element_groups=data_element_group,
            attribute_option_combos=attribute_option_combo,
            include_children=children,
        )
        output_file = f"{output_dir}/data_value_sets.csv"

    # When using the analytics API, two types of requests can be performed:
    # aggregated analytics tables, and raw analytics tables.
    elif mode.lower() == "analytics":

        csv = dhis.analytics(
            periods=period,
            start_date=start,
            end_date=end,
            org_units=org_unit,
            org_unit_groups=org_unit_group,
            org_unit_levels=org_unit_level,
            data_elements=data_element,
            data_element_groups=data_element_group,
            indicators=indicator,
            indicator_groups=indicator_group,
            category_option_combos=category_option_combo,
            programs=program,
        )
        output_file = f"{output_dir}/analytics.csv"

    elif mode.lower() == "analytics-raw":

        csv = dhis.analytics_raw_data(
            periods=period,
            start_date=start,
            end_date=end,
            org_units=org_unit,
            org_unit_groups=org_unit_group,
            org_unit_levels=org_unit_level,
            data_elements=data_element,
            data_element_groups=data_element_group,
            indicators=indicator,
            indicator_groups=indicator_group,
            category_option_combos=category_option_combo,
            programs=program,
        )
        output_file = f"{output_dir}/analytics_raw_data.csv"

    else:
        raise DHIS2ExtractError(f"{mode} is an invalid request mode.")

    with fs.open(output_file, "w") as f:
        logger.debug(f"Writing CSV data to {output_file}.")
        f.write(csv)


# Metadata tables and fields to extract from the DHIS2 instance
METADATA_TABLES = {
    "organisationUnits": "id,code,shortName,name,path,level,geometry",
    "organisationUnitGroups": "id,code,shortName,name,organisationUnits",
    "dataElements": "id,code,shortName,name,aggregationType,zeroIsSignificant",
    "dataElementGroups": "id,code,shortName,name,dataElements",
    "indicators": "id,code,shortName,name,numerator,denominator,annualized",
    "indicatorGroups": "id,name,indicators",
    "dataSets": "id,code,shortName,name,periodType,dataSetElements,organisationUnits,indicators",
    "programs": "id,shortName,name",
    "categoryCombos": "id,code,name,dataDimensionType,categories",
    "categoryOptions": "id,code,shortName,name",
    "categories": "id,code,name,dataDimension",
    "categoryOptionCombos": "id,code,name,categoryCombo,categoryOptions",
}


class DHIS2:
    def __init__(self, instance: str, username: str, password: str, timeout=30):
        """Connect to a DHIS2 instance API.

        Parameters
        ----------
        instance : str
            DHIS2 instance URL.
        username : str
            DHIS2 instance username.
        password : str
            DHIS2 instance password.
        timeout : int
            Default timeout for API calls (in seconds).
        """
        self.api = Api(
            instance,
            username,
            password,
            user_agent="openhexa-pipelines/dhis2-extraction",
        )
        self.timeout = timeout
        self.metadata = self.get_metadata()

    def get_metadata(self) -> dict:
        """Pull main metadata tables from the instance.

        Return a dict with metadata for the following types:
            - organisationUnits
            - organisationUnitGroups
            - dataElements
            - dataElementGroups
            - indicators
            - indicatorGroups
            - dataSets
            - programs
            - categoryCombos
            - categoryOptions
            - categories
            - categoryOptionCombos
        """
        meta = {}
        for name, fields in METADATA_TABLES.items():
            params = {name: True, f"{name}:fields": fields}
            r = self.api.get("metadata", params=params, timeout=self.timeout)
            meta[name] = r.json().get(name)
        return meta

    def data_element_dataset(self, data_element_uid: str) -> str:
        """Identify the dataset of a data element.

        Parameters
        ----------
        data_element_uid : str
            Data element UID.

        Return
        ------
        str
            Dataset UID.
        """
        for dataset in self.metadata.get("dataSets"):
            for data_element in dataset.get("dataSetElements"):
                uid = data_element["dataElement"]["id"]
                if data_element_uid == uid:
                    return dataset.get("id")
        return None

    def org_units_per_lvl(self, level: int) -> typing.Sequence[str]:
        """List org units from a given hierarchical level.

        Parameters
        ----------
        level : int
            Hierarchical level of interest.

        Return
        ------
        list of str
            UIDs of the org units belonging to the provided level.
        """
        return [
            org_unit["id"]
            for org_unit in self.metadata.get("organisationUnits")
            if org_unit.get("level") == level
        ]

    def data_value_sets(
        self,
        datasets: typing.Sequence[str] = None,
        periods: typing.Sequence[str] = None,
        start_date: str = None,
        end_date: str = None,
        org_units: typing.Sequence[str] = None,
        org_unit_groups: typing.Sequence[str] = None,
        org_unit_levels: typing.Sequence[int] = None,
        data_elements: typing.Sequence[str] = None,
        data_element_groups: typing.Sequence[str] = None,
        attribute_option_combos: typing.Sequence[str] = None,
        include_children: bool = False,
    ) -> str:
        """Extract all data values for a given DHIS2 dataset.

        No aggregation. Uses the `dataValueSets` endpoint of the DHIS2 API.
        This request allows to extract data that is not yet stored in the
        analytics table.

        Required parameters:
            * Either `org_units` or `org_unit_groups`
            * Either `periods` or `start_date` and `end_date`
            * Either `datasets` or `data_element_groups`

        Parameters
        ----------
        datasets : list of str, optional
            UID of the datasets of interest.
        periods : list of str, optional
            Periods of interest in DHIS2 format.
        start_date : str, optional
            Start date in ISO format.
        end_date : str, optional
            End date in ISO format.
        org_units : list of str, optional
            UIDs of the organisation units of interest.
        org_unit_groups : list of str, optional
            UIDs of the organisation unit groups of interest.
        org_unit_levels : list of int, optional
            Org unit hierarchical levels of interest.
        data_elements : list of str, optional
            UIDs of the data elements of interest.
        data_element_groups : list of str, optional
            UIDs of the data element groups of interest.
        attribute_option_combos : list of str, optional
            UIDs of the attribute option combos of interest.
        include_children : bool, optional
            Include children of selected org units.

        Return
        ------
        str
            Output data in CSV format.
        """
        if not org_units and not org_unit_groups and not org_unit_levels:
            raise DHIS2ExtractError(
                "Either org unit, org unit group or levels is required."
            )
        if not periods and (not start_date or not end_date):
            raise DHIS2ExtractError("Either period or start/end date is required.")
        if not datasets and not data_element_groups and not data_elements:
            raise DHIS2ExtractError(
                "Either dataset, data element or data element group is required."
            )

        if start_date and end_date:
            if not _check_iso_date(start_date):
                raise DHIS2ExtractError("Start date is not in ISO format.")
            if not _check_iso_date(end_date):
                raise DHIS2ExtractError("End date is not in ISO format.")

        # The dataValueSets endpoint does not support dataElements as a parameter,
        # only dataSets. Here, we get the list of parent datasets required to cover
        # the provided data elements.
        # NB: As a result, the response will contain data elements that have not been
        # requested.
        # NB2: datasets parameter still takes precedence.
        if not datasets and data_elements:
            datasets = [self.data_element_dataset(de) for de in data_elements]
            datasets = list(set(datasets))

        params = {
            "dataSet": datasets,
            "children": include_children,
        }
        if periods:
            params["period"] = periods
        if start_date and end_date:
            params["startDate"] = start_date
            params["endDate"] = end_date
        if org_unit_groups:
            params["orgUnitGroup"] = org_unit_groups
        if data_element_groups:
            params["dataElementGroup"] = data_element_groups
        if attribute_option_combos:
            params["attributeOptionCombo"] = attribute_option_combos

        # The dataValueSets endpoint does not support the "ou:LEVEL-n" dimension
        # parameter. As an alternative, we request data for the parent org units
        # and include data for the children org units in the response.
        # NB: org_units parameter still takes precedence.
        if not org_units and org_unit_levels:
            org_units = []
            for lvl in org_unit_levels:
                org_units += self.org_units_per_lvl(lvl - 1)

            params["children"] = True

            logger.info(
                f"No org units, so asking data for org unit of levels {','.join([str(lvl - 1) for lvl in org_unit_levels])} (total: {len(org_units)})."
            )

        return self.api.chunked_get(
            "dataValueSets",
            params=params,
            chunk_on=("orgUnit", org_units),
            chunk_size=50,
            file_type="csv",
            timeout=self.timeout,
        ).content.decode()

    def analytics_raw_data(
        self,
        periods: typing.Sequence[str] = None,
        start_date: str = None,
        end_date: str = None,
        org_units: typing.Sequence[str] = None,
        org_unit_groups: typing.Sequence[str] = None,
        org_unit_levels: typing.Sequence[int] = None,
        data_elements: typing.Sequence[str] = None,
        data_element_groups: typing.Sequence[str] = None,
        indicators: typing.Sequence[str] = None,
        indicator_groups: typing.Sequence[str] = None,
        category_option_combos: typing.Sequence[str] = None,
        programs: typing.Sequence[str] = None,
    ) -> str:
        """Extract non-aggregated raw data stored in the analytics data tables.

        No aggregation is performed. The response will contain data related to all
        the children of the provided org units.

        Parameters
        ----------
        periods : list of str, optional
            Periods of interest in DHIS2 format.
        start_date : str
            Starting date in ISO format.
        end_date : str
            Ending date in ISO format.
        org_units : list of str, optional
            UIDs of the organisation units of interest.
        org_unit_groups : list of str, optional
            UIDs of the organisation unit groups of interest.
        org_unit_levels : list of int, optional
            Hierarchical org unit levels of interest.
        data_elements : list of str, optional
            UIDs of the data elements of interest.
        data_element_groups : list of str, optional
            UIDs of the data element groups of interest.
        indicators : list of str, optional
            UIDs of the indicators of interest.
        indicator_groups : list of str, optional
            UIDs of the indicator groups of interest.
        category_option_combos : list of str, optional
            UIDs of the category option combos of interest.
        programs : list of str, optional
            UIDs of the programs of interest.

        Return
        ------
        str
            Output data in CSV format.
        """
        if not periods and (not start_date or not end_date):
            raise DHIS2ExtractError("A date range or a period must be provided.")
        if not org_units and not org_unit_groups and not org_unit_levels:
            raise DHIS2ExtractError(
                "At least one org unit or org unit group must be provided."
            )
        if (
            not data_elements
            and not data_element_groups
            and not indicators
            and not indicator_groups
        ):
            raise DHIS2ExtractError(
                "At least on data element or indicator should be provided."
            )
        if start_date and end_date:
            if not _check_iso_date(start_date):
                raise DHIS2ExtractError("Start date is not in ISO format.")
            if not _check_iso_date(end_date):
                raise DHIS2ExtractError("End date is not in ISO format.")

        dimension = _dimension_param(
            periods,
            org_units,
            org_unit_groups,
            org_unit_levels,
            data_elements,
            data_element_groups,
            indicators,
            indicator_groups,
            category_option_combos,
            programs,
        )

        r = self.api.chunked_get(
            "analytics/rawData",
            params={"dimension": None, "startDate": start_date, "endDate": end_date},
            chunk_on=("dimension", self.chunk_dimension_param(dimension)),
            chunk_size=1,
            file_type="csv",
            timeout=self.timeout,
        )

        return r.content.decode()

    def analytics(
        self,
        periods: typing.Sequence[str] = None,
        start_date: str = None,
        end_date: str = None,
        org_units: typing.Sequence[str] = None,
        org_unit_groups: typing.Sequence[str] = None,
        org_unit_levels: typing.Sequence[int] = None,
        data_elements: typing.Sequence[str] = None,
        data_element_groups: typing.Sequence[str] = None,
        indicators: typing.Sequence[str] = None,
        indicator_groups: typing.Sequence[str] = None,
        category_option_combos: typing.Sequence[str] = None,
        programs: typing.Sequence[str] = None,
    ) -> str:
        """Extract aggregated data values from a DHIS2 instance.

        Values are aggretated according to the period and the org unit
        hierarchy. Uses the Analytics endpoint of the DHIS2 API.

        Parameters
        ----------
        periods : list of str, optional
            Periods of interest in DHIS2 format.
        start_date : str, optional
            Start date in DHIS2 format.
        end_date : str, optional
            End date in DHIS2 format.
        org_units : list of str, optional
            UIDs of the organisation units of interest.
        org_unit_groups : list of str, optional
            UIDs of the organisation unit groups of interest.
        org_unit_levels : list of int, optional
            Hierarchical org unit levels of interest.
        data_elements : list of str, optional
            UIDs of the data elements of interest.
        data_element_groups : list of str, optional
            UIDs of the data element groups of interest.
        indicators : list of str, optional
            UIDs of the indicators of interest.
        indicator_groups : list of str, optional
            UIDs of the indicator groups of interest.
        category_option_combos : list of str, optional
            UIDs of the category option combos of interest.
        programs : list of str, optional
            UIDs of the programs of interest.

        Returns
        -------
        str
            Output data in CSV format.
        """
        if not periods and not (start_date and end_date):
            raise DHIS2ExtractError(
                "At least one period or start/end dates must be provided."
            )
        if not org_units and not org_unit_groups and not org_unit_levels:
            raise DHIS2ExtractError(
                "At least one org unit or org unit group must be provided."
            )
        if (
            not data_elements
            and not data_element_groups
            and not indicators
            and not indicator_groups
        ):
            raise DHIS2ExtractError(
                "At least one data element or indicator should be provided."
            )
        if (start_date and not _check_dhis2_period(start_date)) or (
            end_date and not _check_dhis2_period(end_date)
        ):
            raise DHIS2ExtractError(
                "Start and end dates must be in DHIS2 period format."
            )

        # The analytics API doesn't support start and end dates, only periods.
        # Here, start and end dates are assumed to be in the DHIS2 format and
        # are split into a period range according to the format identified, e.g.
        # yearly, monthly, quarterly, etc.
        # NB: The periods parameter still takes precedence over start and end dates,
        # for consistency with the DHIS2 API.
        if not periods and (start_date and end_date):
            periods = get_range(Period(start_date), Period(end_date))

        dimension = _dimension_param(
            periods,
            org_units,
            org_unit_groups,
            org_unit_levels,
            data_elements,
            data_element_groups,
            indicators,
            indicator_groups,
            category_option_combos,
            programs,
            add_empty_co_arg=False,
        )

        r = self.api.chunked_get(
            "analytics",
            params={"dimension": None, "startDate": start_date, "endDate": end_date},
            chunk_on=("dimension", self.chunk_dimension_param(dimension)),
            chunk_size=1,
            file_type="csv",
            timeout=self.timeout,
        )

        return r.content.decode()

    def chunk_dimension_param(
        self, src_dimension_param: typing.List[str], chunk_size: int = 50
    ):
        """Create chunks from dimension params if needed.

        If the "ou:", "dx:" or "pe:" syntax refer to too many elements, split
        the request into multiple chunks of of max length <chunk_size>. Also
        supports the "ou:LEVEL-" syntax.
        """
        ou_params = []
        pe_params = []
        dx_params = []
        other_params = []

        for param in src_dimension_param:

            if param.startswith("ou:"):
                if "LEVEL" in param:
                    level = int(param[-1])
                    org_units = self.org_units_per_lvl(level)
                else:
                    org_units = param[3:].split(";")
                for i in range(0, len(org_units), chunk_size):
                    ou_params.append(
                        f"ou:{';'.join([ou for ou in org_units[i:i+chunk_size]])}"
                    )

            elif param.startswith("dx:"):
                data_elements = param[3:].split(";")
                for i in range(0, len(data_elements), chunk_size):
                    dx_params.append(
                        f"dx:{';'.join([dx for dx in data_elements[i:i+chunk_size]])}"
                    )

            elif param.startswith("pe:"):
                periods = param[3:].split(";")
                for i in range(0, len(periods), chunk_size):
                    pe_params.append(
                        f"pe:{';'.join([pe for pe in periods[i:i+chunk_size]])}"
                    )

            else:
                other_params.append(param)

        chunks = []
        if pe_params:
            for ou_param, dx_param, pe_param in product(
                ou_params, dx_params, pe_params
            ):
                chunks.append(other_params + [ou_param, dx_param, pe_param])
        else:
            for ou_param, dx_param in product(ou_params, dx_params):
                chunks.append(other_params + [ou_param, dx_param])

        return chunks


def _dimension_param(
    periods: typing.Sequence[str] = None,
    org_units: typing.Sequence[str] = None,
    org_unit_groups: typing.Sequence[str] = None,
    org_unit_levels: typing.Sequence[int] = None,
    data_elements: typing.Sequence[str] = None,
    data_element_groups: typing.Sequence[str] = None,
    indicators: typing.Sequence[str] = None,
    indicator_groups: typing.Sequence[str] = None,
    category_option_combos: typing.Sequence[str] = None,
    programs: typing.Sequence[str] = None,
    add_empty_co_arg: bool = True,
) -> typing.Sequence[str]:
    """Format dimension API parameter.

    See the DHIS2 docs for more info:
    <https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-236/analytics.html#webapi_analytics_dimensions_and_items>
    <https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-236/analytics.html#webapi_analytics_dx_dimension>
    """
    dimension = []
    if periods:
        dimension.append("pe:" + ";".join(periods))
    if org_units:
        dimension.append("ou:" + ";".join(org_units))
    if org_unit_groups:
        dimension.append(
            "ou:" + ";".join([f"OU_GROUP-{uid}" for uid in org_unit_groups])
        )
    if org_unit_levels:
        dimension.append("ou:" + ";".join([f"LEVEL-{lvl}" for lvl in org_unit_levels]))
    if data_elements:
        dimension.append("dx:" + ";".join(data_elements))
    if data_element_groups:
        dimension.append(
            "dx:" + ";".join([f"DE_GROUP-{uid}" for uid in data_element_groups])
        )
    if indicators:
        dimension.append("dx:" + ";".join(indicators))
    if indicator_groups:
        dimension.append(
            "dx:" + ";".join([f"IN_GROUP-{uid}" for uid in indicator_groups])
        )
    if programs:
        dimension.append("dx:" + ";".join(programs))
    if category_option_combos:
        dimension.append("co:" + ";".join(category_option_combos))
    elif add_empty_co_arg:
        # add at least an empty coc argument to get COC UIDs in output
        dimension.append("co:")
    return dimension


def _check_iso_date(date: str) -> bool:
    """Check that date string is in ISO format."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date))


def _check_dhis2_period(date: str) -> bool:
    """Check that date string is in DHIS2 format."""
    try:
        Period(date)
        return True
    except NotImplementedError:
        return False


@cli.command()
@click.option("--input-dir", "-i", type=str, help="Input directory.")
@click.option("--output-dir", "-o", type=str, help="Output directory.")
@click.option(
    "--empty-rows", is_flag=True, default=False, help="Empty rows for missing data."
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files."
)
def transform(input_dir, output_dir, empty_rows, overwrite):
    """Transform raw data from DHIS2 into formatted CSV files."""
    output_dir = output_dir.rstrip("/")
    metadata_output_dir = f"{output_dir}/metadata"
    input_dir = input_dir.rstrip("/")
    fs_input = filesystem(input_dir)
    fs_output = filesystem(output_dir)

    # S3FileSystem.mkdirs() will automatically create a new
    # bucket if permissions allow it and we do not want that.
    if output_dir.startswith("s3://"):
        bucket = output_dir.split("//")[-1].split("/")[0]
        if not _s3_bucket_exists(fs_output, bucket):
            raise DHIS2ExtractError(f"S3 bucket {bucket} does not exist.")

    fs_output.mkdirs(output_dir, exist_ok=True)
    fs_output.mkdirs(metadata_output_dir, exist_ok=True)

    fpath_metadata = f"{input_dir}/metadata.json"
    if not fs_input.exists(fpath_metadata):
        raise DHIS2ExtractError(f"Metadata file not found at {fpath_metadata}.")

    with fs_input.open(fpath_metadata) as f:
        metadata = json.load(f)

    # Output file names and corresponding transform function
    # for each metadata table.
    transform_functions = [
        ("organisation_units.csv", _transform_org_units),
        ("organisation_unit_groups.csv", _transform_org_unit_groups),
        ("data_elements.csv", _transform_data_elements),
        ("data_element_groups.csv", _transform_data_element_groups),
        ("indicators.csv", _transform_indicators),
        ("indicator_groups.csv", _transform_indicator_groups),
        ("datasets.csv", _transform_datasets),
        ("programs.csv", _transform_programs),
        ("category_option_combos.csv", _transform_category_option_combos),
        ("category_options.csv", _transform_cat_options),
        ("category_combos.csv", _transform_cat_combos),
        ("categories.csv", _transform_categories),
    ]

    for fname, transform in transform_functions:

        fpath = f"{metadata_output_dir}/{fname}"

        if fs_output.exists(fpath) and not overwrite:
            raise FileExistsError(f"File {fpath} already exists.")

        # Transform metadata and write output dataframe to disk
        logger.info(f"Creating metadata table {fname}.")
        df = transform(metadata)
        with fs_output.open(fpath, "w") as f:
            df.to_csv(f, index=False)

    # Create a GPKG with all org units for which we have geometries
    fpath = f"{metadata_output_dir}/organisation_units.gpkg"
    if fs_output.exists(fpath) and not overwrite:
        raise FileExistsError(f"File {fpath} already exists.")

    logger.info("Creating org units geopackage.")
    fpath = f"{metadata_output_dir}/organisation_units.gpkg"
    if fs_output.exists(fpath) and not overwrite:
        raise FileExistsError(f"File {fpath} already exists.")

    with fs_output.open(f"{metadata_output_dir}/organisation_units.csv") as f:
        df = pd.read_csv(f)
        geodf = _transform_org_units_geo(df)

    # Multi-layered write with the Geopackage driver does not seem to work
    # correctly in Geopandas when using a file handle, that is why we do
    # not use fs_output.open() here. I don't know why it only works with
    # a file path
    with tempfile.NamedTemporaryFile() as tmpf:
        for level in sorted(geodf.ou_level.unique()):
            geodf_lvl = geodf[geodf.ou_level == level]
            geodf_lvl.to_file(
                f"{tmpf.name}.gpkg", driver="GPKG", layer=f"LEVEL_{level}"
            )
        fs_output.put(f"{tmpf.name}.gpkg", fpath)

    # These metadata tables are needed to join element names and full org unit
    # hierarchy into the final extract.
    org_units = pd.read_csv(
        f"{metadata_output_dir}/organisation_units.csv", index_col=0
    )
    data_elements = pd.read_csv(f"{metadata_output_dir}/data_elements.csv", index_col=0)
    indicators = pd.read_csv(f"{metadata_output_dir}/indicators.csv", index_col=0)
    coc = pd.read_csv(f"{metadata_output_dir}/category_option_combos.csv", index_col=0)

    # Transform API response
    for fname in ["analytics.csv", "analytics_raw_data.csv", "data_value_sets.csv"]:

        fpath_input = f"{input_dir}/{fname}"
        fpath_output = f"{output_dir}/extract.csv"

        if not fs_input.exists(fpath_input):
            continue

        if fs_output.exists(fpath_output) and not overwrite:
            raise FileExistsError(f"File {fpath_output} already exists.")

        logger.info(f"Processing API response {fpath_input}.")

        with fs_input.open(fpath_input) as f:
            api_response = pd.read_csv(f)
        extract = _transform(api_response)

        if empty_rows:
            extract = _add_empty_rows(
                extract, index_columns=["dx_uid", "coc_uid", "period", "ou_uid"]
            )

        extract = _join_from_metadata(
            extract, data_elements, indicators, coc, org_units
        )

        for column in extract.columns:
            if column.startswith("Unnamed"):
                extract.drop(columns=column, inplace=True)

        extract.dropna(axis=1, how="all", inplace=True)

        with fs_output.open(fpath_output, "w") as f:
            extract.to_csv(f, index=False)


def _transform_org_units(metadata: dict) -> pd.DataFrame:
    """Transform org units metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("organisationUnits"))
    df = df[["id", "code", "shortName", "name", "path", "geometry"]]
    df.columns = ["ou_uid", "ou_code", "ou_shortname", "ou_name", "path", "geometry"]
    df["ou_level"] = df.path.apply(lambda x: x.count("/"))
    df = df[
        ["ou_uid", "ou_code", "ou_shortname", "ou_name", "ou_level", "path", "geometry"]
    ]  # Reorder columns
    return df


def _transform_org_units_geo(org_units: pd.DataFrame) -> gpd.GeoDataFrame:
    """Transform org units metadata dataframe into a GeoDataFrame."""
    geodf = gpd.GeoDataFrame(
        org_units,
        crs="epsg:4326",
        geometry=[
            shape(json.loads(geom.replace("'", '"'))) if not pd.isna(geom) else None
            for geom in org_units.geometry
        ],
    )
    return geodf


def _transform_org_unit_groups(metadata: dict) -> pd.DataFrame:
    """Transform org unit groups metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("organisationUnitGroups"))
    df = df[["id", "code", "shortName", "name", "organisationUnits"]]
    df.columns = ["oug_uid", "oug_code", "oug_shortname", "oug_name", "org_units"]
    df["org_units"] = df.org_units.apply(lambda x: ";".join(ou.get("id") for ou in x))
    return df


def _transform_data_elements(metadata: dict) -> pd.DataFrame:
    """Transform data elements metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("dataElements"))
    df = df[["id", "code", "shortName", "name", "aggregationType", "zeroIsSignificant"]]
    df.columns = [
        "dx_uid",
        "dx_code",
        "dx_shortname",
        "dx_name",
        "aggregation_type",
        "zero_is_significant",
    ]
    return df


def _transform_data_element_groups(metadata: dict) -> pd.DataFrame:
    """Transform data element groups metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("dataElementGroups"))
    df = df[["id", "name", "dataElements"]]
    df.columns = ["deg_uid", "deg_name", "data_elements"]
    df["data_elements"] = df.data_elements.apply(
        lambda x: ";".join(de.get("id") for de in x)
    )
    return df


def _transform_indicators(metadata: dict) -> pd.DataFrame:
    """Transform indicators metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("indicators"))
    df = df[
        ["id", "code", "shortName", "name", "numerator", "denominator", "annualized"]
    ]
    df.columns = [
        "dx_uid",
        "dx_code",
        "dx_shortname",
        "dx_name",
        "numerator",
        "denominator",
        "annualized",
    ]
    return df


def _transform_indicator_groups(metadata: dict) -> pd.DataFrame:
    """Transform indicator groups metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("indicatorGroups"))
    df = df[["id", "name", "indicators"]]
    df.columns = ["ing_uid", "ing_name", "indicators"]
    df["indicators"] = df.indicators.apply(
        lambda x: ";".join(indicator.get("id") for indicator in x)
    )
    return df


def _transform_datasets(metadata: dict) -> pd.DataFrame:
    """Transform datasets metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("dataSets"))
    df = df[
        [
            "id",
            "code",
            "shortName",
            "name",
            "periodType",
            "dataSetElements",
            "organisationUnits",
            "indicators",
        ]
    ]
    df.columns = [
        "ds_uid",
        "ds_code",
        "ds_short_name",
        "ds_name",
        "period_type",
        "data_elements",
        "org_units",
        "indicators",
    ]
    df["data_elements"] = df.data_elements.apply(
        lambda x: ";".join([dx.get("dataElement").get("id") for dx in x])
    )
    df["indicators"] = df.indicators.apply(
        lambda x: ";".join([dx.get("id") for dx in x])
    )
    df["org_units"] = df.org_units.apply(lambda x: ";".join([ou.get("id") for ou in x]))
    return df


def _transform_programs(metadata: dict) -> pd.DataFrame:
    """Transform programs metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("programs"))
    df = df[["id", "shortName", "name"]]
    df.columns = ["program_uid", "program_shortname", "program_name"]
    return df


def _transform_cat_combos(metadata: dict) -> pd.DataFrame:
    """Transform category combos metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryCombos"))
    df = df[["id", "code", "name", "dataDimensionType", "categories"]]
    df.columns = ["cc_uid", "cc_code", "cc_name", "data_dimension_type", "categories"]
    df["categories"] = df.categories.apply(
        lambda x: ";".join([cc.get("id") for cc in x])
    )
    return df


def _transform_cat_options(metadata: dict) -> pd.DataFrame:
    """Transform category options metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryOptions"))
    df = df[["id", "code", "shortName", "name"]]
    df.columns = ["co_uid", "co_code", "co_shortname", "co_name"]
    return df


def _transform_categories(metadata: dict) -> pd.DataFrame:
    """Transform categories metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categories"))
    df = df[["id", "code", "name", "dataDimension"]]
    df.columns = ["cat_uid", "cat_code", "cat_name", "data_dimension"]
    return df


def _transform_category_option_combos(metadata: dict) -> pd.DataFrame:
    """Transform category option combos into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryOptionCombos"))
    df = df[["id", "code", "name", "categoryCombo", "categoryOptions"]]
    df.columns = [
        "coc_uid",
        "coc_code",
        "coc_name",
        "category_combo",
        "category_options",
    ]
    df["category_combo"] = df.category_combo.apply(lambda x: x.get("id"))
    df["category_options"] = df.category_options.apply(
        lambda x: ";".join([co.get("id") for co in x])
    )
    return df


def _transform(data: pd.DataFrame) -> pd.DataFrame:
    """Transform API response into a formatted dataframe."""
    COLUMNS = {
        "Data": "dx_uid",
        "dataelement": "dx_uid",
        "Category option combo": "coc_uid",
        "categoryoptioncombo": "coc_uid",
        "Period": "period",
        "Unnamed: 3": "period",
        "period": "period",
        "Organisation unit": "ou_uid",
        "orgunit": "ou_uid",
        "Value": "value",
        "value": "value",
        "lastupdated": "last_updated",
    }
    columns_in_data = {src: dst for src, dst in COLUMNS.items() if src in data.columns}
    df = data[[col for col in columns_in_data]]
    df = data.rename(columns=columns_in_data)
    return df


def _add_empty_rows(
    dataframe: pd.DataFrame, index_columns: typing.List[str]
) -> pd.DataFrame:
    """Create new rows with NaN values as missing data.

    A MultiIndex is generated for the input dataframe based on the
    column names identified in `index_columns`. In the output
    dataframe, a new row with NaN values is added for each combination
    of index column (e.g. dx_uid + coc_uid + ou_uid + period) without
    any data.
    """
    index_columns = [c for c in index_columns if c in dataframe]
    multi_index = pd.MultiIndex.from_product(
        iterables=[dataframe[column].unique() for column in index_columns],
        names=index_columns,
    )
    dataframe_ = dataframe.set_index(index_columns)
    nan_filled = pd.DataFrame(
        index=multi_index,
        columns=[col for col in dataframe.columns if col not in index_columns],
    )

    for idx in nan_filled.index:
        if idx in dataframe_.index:
            nan_filled.loc[idx] = dataframe_.loc[idx]

    return nan_filled.reset_index(drop=False)


def _dx_type(dx_uid: str, data_elements: pd.DataFrame, indicators: pd.DataFrame) -> str:
    """Get the data type corresponding to an UID (data element or indicator).

    Examples
    --------
    >>> _dx_type("vI2csg55S9C", data_elements, indicators)
    'Data Element'
    """
    if dx_uid in data_elements.index:
        return "Data Element"
    elif dx_uid in indicators.index:
        return "Indicator"
    else:
        raise DHIS2ExtractError(
            f"DX {dx_uid} not found in data elements or indicators metadata."
        )


def _dx_name(dx_uid: str, data_elements: pd.DataFrame, indicators: pd.DataFrame) -> str:
    """Get the name of a data element or indicator.

    Examples
    --------
    >>> _dx_name("vI2csg55S9C", data_elements, indicators)
    'OPV3 doses given'
    """
    if _dx_type(dx_uid, data_elements, indicators) == "Data Element":
        return data_elements.at[dx_uid, "dx_name"]
    elif _dx_type(dx_uid, data_elements, indicators) == "Indicator":
        return indicators.at[dx_uid, "dx_name"]
    else:
        raise DHIS2ExtractError(
            f"DX {dx_uid} not found in data elements or indicators metadata."
        )


def _level_uid(ou_path: str, level: int) -> str:
    """Get the org unit UID corresponding to a given hierarchical level.

    Examples
    --------
    >>> _level_uid("/ImspTQPwCqd/O6uvpzGd5pu/vWbkYPRmKyS", 2)
    'O6uvpzGd5pu'
    """
    hierarchy = [ou for ou in ou_path.split("/") if ou]
    if level <= len(hierarchy):
        return hierarchy[level - 1]
    else:
        return None


def _join_from_metadata(
    extract: pd.DataFrame,
    data_elements: pd.DataFrame,
    indicators: pd.DataFrame,
    category_option_combos: pd.DataFrame,
    organisation_units: pd.DataFrame,
) -> pd.DataFrame:
    """Join fields from the metadata tables into the extract.

    More specifically, the following columns are added to the dataframe:
        * dx_name (name of the data element/indicator)
        * dx_type ('Data Element' or 'Indicator')
        * coc_name (name of the category option combo)
        * level_<lvl>_uid (UID of the org unit for each hierarchical level)
        * level_<lvl>_name (name of the org unit for each hierarchical level)
    """
    extract = extract.copy()
    extract["dx_name"] = extract.dx_uid.apply(
        lambda uid: _dx_name(uid, data_elements, indicators)
    )
    extract["dx_type"] = extract.dx_uid.apply(
        lambda uid: _dx_type(uid, data_elements, indicators)
    )

    # in some cases the extract does not contain any COC info
    if "coc_uid" in extract:
        extract["coc_name"] = extract.coc_uid.apply(
            lambda uid: category_option_combos.at[uid, "coc_name"]
        )

    # Max number of hierarchical levels in the instance
    levels = len(organisation_units.path.max().split("/")) - 1

    # Add UID and name for each hierarchical level
    for level in range(1, levels + 1):
        column_uid = f"level_{level}_uid"
        column_name = f"level_{level}_name"
        extract[column_uid] = extract.ou_uid.apply(
            lambda uid: _level_uid(organisation_units.at[uid, "path"], level)
        )
        extract[column_name] = extract[column_uid].apply(
            lambda uid: organisation_units.at[uid, "ou_name"] if uid else None
        )

    return extract


if __name__ == "__main__":
    cli()
