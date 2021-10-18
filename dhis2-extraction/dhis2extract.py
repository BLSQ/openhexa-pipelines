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
        self.api = Api(instance, username, password)
        self.metadata = self.api.get("metadata", timeout=timeout).json()
        self.timeout = timeout

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

    # metadata
    filepath = f"{input_dir}/metadata.json"
    if fs_input.exists(filepath):

        with fs_input.open(filepath) as f:
            metadata = json.load(f)

        # org units
        output_file = f"{output_dir}/organisation_units.csv"
        df = _transform_org_units(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # org unit geometries
        output_file = f"{output_dir}/organisation_units.gpkg"
        geodf = _transform_org_units_geo(df)
        with fs_output.open(output_file, "w") as f:
            geodf.to_file(output_file, driver="GPKG")

        # org unit groups
        output_file = f"{output_dir}/organisation_unit_groups.csv"
        df = _transform_org_unit_groups(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # data elements
        output_file = f"{output_dir}/data_elements.csv"
        df = _transform_data_elements(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # datasets
        output_file = f"{output_dir}/datasets.csv"
        df = _transform_datasets(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # category option combos
        output_file = f"{output_dir}/category_option_combos.csv"
        df = _transform_coc(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # category combos
        output_file = f"{output_dir}/category_combos.csv"
        df = _transform_cat_combos(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # category options
        output_file = f"{output_dir}/category_options.csv"
        df = _transform_cat_options(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

        # categories
        output_file = f"{output_dir}/categories.csv"
        df = _transform_categories(metadata)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

    # analytics API output
    filepath = f"{input_dir}/analytics.csv"
    if fs_input.exists(filepath):
        output_file = f"{output_dir}/extract.csv"
        with fs_input.open(filepath) as f:
            df_raw = pd.read_csv(f)
        df = _transform_analytics(df_raw)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

    # analytics/rawData API output
    filepath = f"{input_dir}/analytics_raw_data.csv"
    if fs_input.exists(filepath):
        output_file = f"{output_dir}/extract.csv"
        with fs_input.open(filepath) as f:
            df_raw = pd.read_csv(f)
        df = _transform_analytics_raw_data(df_raw)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)

    # dataValueSet API output
    filepath = f"{input_dir}/data_value_sets.csv"
    if fs_input.exists(filepath):
        output_file = f"{output_dir}/extract.csv"
        with fs_input.open(filepath) as f:
            df_raw = pd.read_csv(f)
        df = _transform_data_value_sets(df_raw)
        with fs_output.open(output_file, "w") as f:
            df.to_csv(f, index=False)


def _transform_org_units(metadata: dict) -> pd.DataFrame:
    """Transform org units metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("organisationUnits"))
    df = df[["id", "code", "shortName", "name", "path", "geometry"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "NAME", "PATH", "GEOMETRY"]
    return df


def _transform_org_units_geo(org_units: pd.DataFrame) -> gpd.GeoDataFrame:
    """Transform org units metadata dataframe into a GeoDataFrame."""
    geodf = gpd.GeoDataFrame(
        org_units,
        crs="epsg:4326",
        geometry=[
            shape(geom) if not pd.isna(geom) else None for geom in org_units.GEOMETRY
        ],
    )
    geodf = geodf.drop(["GEOMETRY"], axis=1)
    return geodf


def _transform_org_unit_groups(metadata: dict) -> pd.DataFrame:
    """Transform org unit groups metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("organisationUnitGroups"))
    df = df[["id", "code", "shortName", "name", "organisationUnits"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "NAME", "ORG_UNITS"]
    df["ORG_UNITS"] = df.ORG_UNITS.apply(lambda x: ";".join(ou.get("id") for ou in x))
    return df


def _transform_data_elements(metadata: dict) -> pd.DataFrame:
    """Transform data elements metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("dataElements"))
    df = df[["id", "code", "shortName", "name", "aggregationType", "zeroIsSignificant"]]
    df.columns = [
        "UID",
        "CODE",
        "SHORT_NAME",
        "NAME",
        "AGGREGATION_TYPE",
        "ZERO_IS_SIGNIFICANT",
    ]
    return df


def _transform_datasets(metadata: dict) -> pd.DataFrame:
    """Transform datasets metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("dataSets"))
    df = df[["id", "code", "shortName", "name"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "NAME"]
    return df


def _transform_coc(metadata: dict) -> pd.DataFrame:
    """Transform category option combos metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryOptionCombos"))
    df = df[["id", "code", "name", "categoryCombo", "categoryOptions"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "CATEGORY_COMBO", "CATEGORY_OPTIONS"]
    df["CATEGORY_COMBO"] = df.CATEGORY_COMBO.apply(lambda x: x.get("id"))
    df["CATEGORY_OPTIONS"] = df.CATEGORY_OPTIONS.apply(
        lambda x: ";".join([co.get("id") for co in x])
    )
    return df


def _transform_cat_combos(metadata: dict) -> pd.DataFrame:
    """Transform category combos metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryCombos"))
    df = df[["id", "code", "name", "dataDimensionType", "categories"]]
    df.columns = ["UID", "CODE", "NAME", "DATA_DIMENSION_TYPE", "CATEGORIES"]
    df["CATEGORIES"] = df.CATEGORIES.apply(
        lambda x: ";".join([df.get("id") for df in x])
    )
    return df


def _transform_cat_options(metadata: dict) -> pd.DataFrame:
    """Transform category options metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categoryOptions"))
    df = df[["id", "code", "shortName", "name"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "NAME"]
    return df


def _transform_categories(metadata: dict) -> pd.DataFrame:
    """Transform categories metadata into a formatted DataFrame."""
    df = pd.DataFrame.from_dict(metadata.get("categories"))
    df = df[["id", "code", "shortName", "name", "dataDimension"]]
    df.columns = ["UID", "CODE", "SHORT_NAME", "NAME", "DATA_DIMENSION"]
    return df


def _transform_analytics(data: pd.DataFrame) -> pd.DataFrame:
    """Transform analytics API output into a formatted DataFrame."""
    df = data[["Data", "Category option combo", "Period", "Organisation unit", "Value"]]
    df.columns = ["DX_UID", "COC_UID", "PERIOD", "OU_UID", "VALUE"]
    return df


def _transform_data_value_sets(data: pd.DataFrame) -> pd.DataFrame:
    """Transform dataValueSets API output into a formatted DataFrame."""
    df = data[
        [
            "dataelement",
            "categoryoptioncombo",
            "attributeoptioncombo",
            "period",
            "orgunit",
            "value",
        ]
    ]
    df.columns = ["DX_UID", "COC_UID", "AOC_UID", "PERIOD", "OU_UID", "VALUE"]
    return df


def _transform_analytics_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Transform analytics/rawData API output into a formatted DataFrame."""
    df = data[
        ["Data", "Category option combo", "Unnamed: 3", "Organisation unit", "Value"]
    ]
    df.columns = ["DX_UID", "COC_UID", "PERIOD", "OU_UID", "VALUE"]
    return df


if __name__ == "__main__":
    cli()
