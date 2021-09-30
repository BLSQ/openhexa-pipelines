import json
import logging
import os
import re
import typing

import click
from dhis2 import Api
from fsspec import AbstractFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

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
@click.option("--output", "-o", type=str, required=True, help="Output CSV file.")
@click.option("--output-meta", "-m", type=str, help="Output JSON metadata.")
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
    "--org_unit_level", "-lvl", type=int, multiple=True, help="Organisation unit level."
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
    output: str,
    output_meta: str,
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
    dhis = DHIS2(instance, username, password, timeout=30)

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

    if start and end:
        if not _check_iso_date(start):
            raise DHIS2ExtractError("Start date is not in ISO format.")
        if not _check_iso_date(end):
            raise DHIS2ExtractError("End date is not in ISO format.")

    fs = filesystem(output_meta)
    if output_meta:
        if fs.exists(output_meta) and not overwrite:
            logger.debug("Output metadata file already exists. Skipping.")
        else:
            with fs.open(output_meta) as f:
                logger.debug(f"Writing metadata to {output_meta}.")
                json.dump(dhis.metadata, f)

    if skip:
        return

    # Get the list of all the org unit UIDs corresponding to the selected levels
    if org_unit_level:
        org_unit = []
        for lvl in org_unit_level:
            logger.debug(f"Adding org units of level {lvl}.")
            org_unit.append(*dhis.org_units_per_lvl(lvl))

    # The dataValueSets endpoint does not support data elements UIDs as parameters,
    # only datasets.
    if not analytics:

        csv = dhis.data_value_sets(
            datasets=dataset,
            periods=period,
            start_date=start,
            end_date=end,
            org_units=org_unit,
            org_unit_groups=org_unit,
            data_element_groups=data_element_group,
            attribute_option_combos=attribute_option_combo,
            include_childrens=children,
        )

    # When using the analytics API, two types of requests can be performed:
    # aggregated analytics tables, and raw analytics tables.
    else:

        if aggregate:

            csv = dhis.analytics(
                periods=period,
                org_units=org_unit,
                org_unit_groups=org_unit_group,
                org_unit_levels=org_unit_level,
                data_elements=data_element,
                data_element_groups=data_element_group,
                indicators=indicator,
                indicator_groups=indicator_group,
                programs=program,
            )

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

    fs = filesystem(output)
    if not fs.exists(output) or overwrite:
        with fs.open(output) as f:
            logger.debug(f"Writing CSV data to {output}.")
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
        self.metadata = self.get("metadata", timeout=timeout).json()
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
        datasets: typing.Sequence[str],
        periods: typing.Sequence[str] = None,
        start_date: str = None,
        end_date: str = None,
        org_units: typing.Sequence[str] = None,
        org_unit_groups: typing.Sequence[str] = None,
        data_element_groups: typing.Sequence[str] = None,
        attribute_option_combos: typing.Sequence[str] = None,
        include_childrens: bool = False,
    ) -> str:
        """Extract all data values for a given DHIS2 dataset.

        No aggregation. Uses the `dataValueSets` endpoint of the DHIS2 API.
        This request allows to extract data that is not yet stored in the
        analytics table.

        Parameters
        ----------
        datasets : list of str
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
        if not org_units and not org_unit_groups:
            raise DHIS2ExtractError("No org unit provided.")
        if not periods and (not start_date or not end_date):
            raise DHIS2ExtractError("No period provided.")

        if start_date and end_date:
            if not _check_iso_date(start_date):
                raise DHIS2ExtractError("Start date is not in ISO format.")
            if not _check_iso_date(end_date):
                raise DHIS2ExtractError("End date is not in ISO format.")

        params = {"dataSet": datasets, "children": include_childrens}
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
        periods: typing.Sequence[str],
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
    """Format dimension API parameter."""
    dimension = []
    if periods:
        dimension.append("pe:" + periods.join(";"))
    if org_units:
        dimension.append("ou:" + org_units.join(";"))
    if org_unit_groups:
        dimension.append(
            "ou:" + [f"OU_GROUP-{uid}" for uid in org_unit_groups].join(";")
        )
    if org_unit_levels:
        dimension.append("ou:" + [f"LEVEL-{lvl}" for lvl in org_unit_levels].join(";"))
    if data_elements:
        dimension.append("dx:" + data_elements.join(";"))
    if data_element_groups:
        dimension.append(
            "dx:" + [f"DE_GROUP-{uid}" for uid in data_element_groups].join(";")
        )
    if indicators:
        dimension.append("dx:" + indicators.join(";"))
    if indicator_groups:
        dimension.append(
            "dx:" + [f"IN_GROUP-{uid}" for uid in indicator_groups].join(";")
        )
    if programs:
        dimension.append("dx:" + programs.join(";"))
    return dimension


def _check_iso_date(date: str) -> bool:
    """Check that date string is in ISO format."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date))


if __name__ == "__main__":
    cli()
