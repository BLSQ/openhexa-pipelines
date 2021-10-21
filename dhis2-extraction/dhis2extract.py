import json
import logging
import os
import re
import typing

import click
import geopandas as gpd
import pandas as pd
from dhis2 import Api
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from period import Period, get_range
from s3fs import S3FileSystem
from shapely.geometry import shape

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


@click.group()
def cli():
    pass


@cli.command()
@click.option("--instance", "-i", type=str, required=True, help="DHIS2 instance URL.")
@click.option(
    "--username",
    "-u",
    type=str,
    required=True,
    envvar="DHIS2_USERNAME",
    help="DHIS2 username.",
)
@click.option(
    "--password",
    "-p",
    type=str,
    required=True,
    envvar="DHIS2_PASSWORD",
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
@click.option("--program", "-prg", type=str, multiple=True, help="Program UID.")
@click.option("--from-json", type=str, help="Load parameters from a JSON file.")
@click.option(
    "--children/--no-children",
    is_flag=True,
    default=False,
    help="Include childrens of selected org units.",
)
@click.option(
    "--aggregate/--no-aggregate",
    is_flag=True,
    default=False,
    help="Aggregate using Analytics API.",
)
@click.option(
    "--analytics/--no-analytics",
    is_flag=True,
    default=True,
    help="Use the Analytics API.",
)
@click.option("--skip", is_flag=True, default=False, help="Only download metadata.")
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
    program: typing.Sequence[str],
    from_json: str,
    children: bool,
    aggregate: bool,
    analytics: bool,
    skip: bool,
    overwrite: bool,
):
    """Download data from a DHIS2 instance via its web API."""
    dhis = DHIS2(instance, username, password, timeout=30)
    output_dir = output_dir.rstrip("/")
    fs = filesystem(output_dir)
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
        program = params.get("program", program)

    output_meta = f"{output_dir}/metadata.json"
    if fs.exists(output_meta) and not overwrite:
        logger.debug("Output metadata file already exists. Skipping.")
    else:
        with fs.open(output_meta, "w") as f:
            logger.debug(f"Writing metadata to {output_meta}.")
            json.dump(dhis.metadata, f)

    if skip:
        return

    # The dataValueSets endpoint does not support data elements UIDs as parameters,
    # only datasets.
    if not analytics:

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
            include_childrens=children,
        )
        output_file = f"{output_dir}/data_value_sets.csv"

    # When using the analytics API, two types of requests can be performed:
    # aggregated analytics tables, and raw analytics tables.
    else:

        if aggregate:

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
                programs=program,
            )
            output_file = f"{output_dir}/analytics.csv"

        else:

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
                programs=program,
            )
            output_file = f"{output_dir}/analytics_raw_data.csv"

    if not fs.exists(output_file) or overwrite:
        with fs.open(output_file, "w") as f:
            logger.debug(f"Writing CSV data to {output_file}.")
            f.write(csv)
    else:
        logger.debug("Output CSV file already exists. Skipping.")


# Metadata tables and fields to extract from the DHIS2 instance
METADATA_TABLES = {
    "organisationUnits": "id,code,shortName,name,path,geometry",
    "organisationUnitGroups": "id,code,shortName,name,organisationUnits",
    "dataElements": "id,code,shortName,name,aggregationType,zeroIsSignificant",
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
        org_unit_levels: typing.Sequence[str] = None,
        data_elements: typing.Sequence[str] = None,
        data_element_groups: typing.Sequence[str] = None,
        attribute_option_combos: typing.Sequence[str] = None,
        include_childrens: bool = False,
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
        include_childrens : bool, optional
            Include childrens of selected org units.

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

        # The dataValueSets endpoint does not support the "ou:LEVEL-n" dimension
        # parameter. As an alternative, we request data for the parent org units
        # and include data for the children org units in the response.
        # NB: org_units parameter still takes precedence.
        if not org_units and org_unit_levels:
            org_units = []
            for lvl in org_unit_levels:
                org_units.append(self.org_units_per_lvl(lvl - 1))
            include_childrens = True

        params = {
            "dataSet": datasets,
            "children": include_childrens,
        }
        if periods:
            params["period"] = periods
        if start_date and end_date:
            params["startDate"] = start_date
            params["endDate"] = end_date
        if org_units:
            params["orgUnit"] = org_units
        if org_unit_groups:
            params["orgUnitGroup"] = org_unit_groups
        if data_element_groups:
            params["dataElementGroup"] = data_element_groups
        if attribute_option_combos:
            params["attributeOptionCombo"] = attribute_option_combos

        r = self.api.get(
            "dataValueSets", params=params, file_type="csv", timeout=self.timeout
        )
        return r.content.decode()

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
            programs,
        )

        r = self.api.get(
            "analytics/rawData",
            params={
                "dimension": dimension,
                "startDate": start_date,
                "endDate": end_date,
            },
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
        if not _check_dhis2_period(start_date) or not _check_dhis2_period(end_date):
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
            programs,
        )

        r = self.api.get(
            "analytics",
            params={"dimension": dimension},
            file_type="csv",
            timeout=self.timeout,
        )
        return r.content.decode()


def _dimension_param(
    periods: typing.Sequence[str] = None,
    org_units: typing.Sequence[str] = None,
    org_unit_groups: typing.Sequence[str] = None,
    org_unit_levels: typing.Sequence[int] = None,
    data_elements: typing.Sequence[str] = None,
    data_element_groups: typing.Sequence[str] = None,
    indicators: typing.Sequence[str] = None,
    indicator_groups: typing.Sequence[str] = None,
    programs: typing.Sequence[str] = None,
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
    dimension.append("co:")  # empty to fetch all the category option combos
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
    "--overwrite", is_flag=True, default=False, help="Overwrite existing files."
)
def transform(input_dir, output_dir, overwrite):
    """Transform raw data from DHIS2 into formatted CSV files."""
    output_dir = output_dir.rstrip("/")
    input_dir = input_dir.rstrip("/")
    fs_input = filesystem(input_dir)
    fs_output = filesystem(output_dir)
    fs_output.mkdirs(output_dir, exist_ok=True)

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

        fpath = f"{output_dir}/{fname}"

        # Transform metadata and write output dataframe to disk
        if not fs_output.exists(fpath) or overwrite:
            logger.info(f"Creating metadata table {fname}.")
            df = transform(metadata)
            with fs_output.open(fpath, "w") as f:
                df.to_csv(f, index=False)

        # Skip if output file already exists and --overwrite is not set
        else:
            logger.info(f"{fname} already exists. Skipping.")
            continue

    # Create a GPKG with all org units for which we have geometries
    fpath = f"{output_dir}/organisation_units.gpkg"
    if not fs_output.exists(fpath) or overwrite:
        logger.info("Creating org units geopackage.")
        with fs_output.open(f"{output_dir}/organisation_units.csv") as f:
            df = pd.read_csv(f)
            geodf = _transform_org_units_geo(df)
        with fs_output.open(f"{output_dir}/organisation_units.gpkg", "wb") as f:
            geodf.to_file(f, driver="GPKG")
    else:
        logger.info(f"{os.path.basename(fpath)} already exists. Skipping.")

    # These metadata tables are needed to join element names and full org unit
    # hierarchy into the final extract.
    org_units = pd.read_csv(f"{output_dir}/organisation_units.csv", index_col=0)
    data_elements = pd.read_csv(f"{output_dir}/data_elements.csv", index_col=0)
    indicators = pd.read_csv(f"{output_dir}/indicators.csv", index_col=0)
    coc = pd.read_csv(f"{output_dir}/category_option_combos.csv", index_col=0)

    # Transform API response
    transform_functions = [
        ("analytics.csv", _transform_analytics),
        ("analytics_raw_data.csv", _transform_analytics_raw_data),
        ("data_value_sets", _transform_data_value_sets),
    ]

    for fname, transform in transform_functions:

        fpath_input = f"{input_dir}/{fname}"
        fpath_output = f"{output_dir}/extract.csv"

        if not fs_input.exists(fpath_input):
            continue

        if fs_input.exists(fpath_output) and not overwrite:
            logger.info(f"{os.path.basename(fpath_output)} already exists. Skipping.")
            continue

        logger.info(f"Processing API response {fpath_input}.")

        with fs_input.open(fpath_input) as f:
            api_response = pd.read_csv(f)
        extract = transform(api_response)
        extract = _join_from_metadata(
            extract, data_elements, indicators, coc, org_units
        )

        with fs_output.open(fpath_output, "w") as f:
            extract.to_csv(f, index=False)


def _transform_org_units(metadata: dict) -> pd.DataFrame:
    """Transform org units metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("organisationUnits"))
    df = df[["id", "code", "shortName", "name", "path", "geometry"]]
    df.columns = ["ou_uid", "ou_code", "ou_shortname", "ou_name", "path", "geometry"]
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


def _transform_analytics(data: pd.DataFrame) -> pd.DataFrame:
    """Transform analytics API output into a formatted DataFrame."""
    df = data[["Data", "Category option combo", "Period", "Organisation unit", "Value"]]
    df.columns = ["dx_uid", "coc_uid", "period", "ou_uid", "value"]
    return df


def _transform_data_value_sets(data: pd.DataFrame) -> pd.DataFrame:
    """Transform dataValueSets API output into a formatted DataFrame."""
    df = data[
        [
            "dataelement",
            "categoryoptioncombo",
            "period",
            "orgunit",
            "value",
        ]
    ]
    df.columns = ["dx_uid", "coc_uid", "period", "ou_uid", "value"]
    return df


def _transform_analytics_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Transform analytics/rawData API output into a formatted DataFrame."""
    df = data[
        ["Data", "Category option combo", "Unnamed: 3", "Organisation unit", "Value"]
    ]
    df.columns = ["dx_uid", "coc_uid", "period", "ou_uid", "value"]
    return df


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
