import logging
import os
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


@click.command()
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
@click.option(
    "--start", "-s", type=str, required=True, help="Start date in DHIS2 format."
)
@click.option("--end", "-e", type=str, help="End date in DHIS2 format.")
@click.option(
    "--org-unit", "-ou", type=str, multiple=True, help="Organisation unit UID."
)
@click.option(
    "--org-units-group",
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
@click.option(
    "--indicator", "-in", type=str, multiple=True, help="Indicator group UID."
)
@click.option(
    "--indicator_group", "-ing", type=str, multiple=True, help="Indicator group UID."
)
@click.option(
    "--attribute-option-combo",
    "-aoc",
    type=str,
    multiple=True,
    help="Attribute option combo UID.",
)
@click.option("--program", "-prg", type=str, multiple=True, help="Program UID.")
@click.option(
    "--include-childrens",
    is_flag=True,
    default=False,
    help="Include childrens of selected org units.",
)
@click.option(
    "--aggregate",
    "-agg",
    is_flag=True,
    default=True,
    help="Aggregate using Analytics API.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing file."
)
def extract(
    instance: str,
    username: str,
    password: str,
    output: str,
    period: typing.Sequence[str],
    start: str,
    end: str,
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
    include_childrens: bool,
    aggregate: bool,
    overwrite: bool,
):
    fs = filesystem(output)
    dhis = DHIS2(instance, username, password)

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

        csv = dhis.data_values(
            datasets=dataset,
            periods=period,
            start_date=start,
            end_date=end,
            org_units=org_unit,
            org_unit_groups=org_unit,
            data_element_groups=data_element_group,
            attribute_option_combos=attribute_option_combo,
            include_childrens=include_childrens,
        )

    if not fs.exists(output) or overwrite:
        with fs.open(output) as f:
            f.write(csv)


class DHIS2:
    def __init__(self, instance: str, username: str, password: str):
        """Connect to a DHIS2 instance API.

        Parameters
        ----------
        instance : str
            DHIS2 instance URL.
        username : str
            DHIS2 instance username.
        password : str
            DHIS2 instance password.
        """
        self.api = Api(instance, username, password)

    def data_values(
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
            CSV output table.
        """
        if not org_units and not org_unit_groups:
            raise DHIS2ExtractError("No org unit provided.")
        if not periods and (not start_date or not end_date):
            raise DHIS2ExtractError("No period provided.")

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

        return self.api.get("dataValueSets", params=params, file_type="csv")

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
        data_element : list of str, optional
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

        dimensions = [
            *[f"pe:{pe}" for pe in periods],
            *[f"ou:{ou}" for ou in org_units],
            *[f"ou:OU_GROUP-{oug}" for oug in org_unit_groups],
            *[f"ou:LEVEL-{lvl}" for lvl in org_unit_levels],
            *[f"dx:{de}" for de in data_elements],
            *[f"dx:{ind}" for ind in indicators],
            *[f"dx:DE_GROUP-{deg}" for deg in data_element_groups],
            *[f"dx:IN_GROUP-{ing}" for ing in indicator_groups],
            *[f"dx:{prg}" for prg in programs],
        ]

        return self.api.get(
            "analytics.csv", params={"dimension": dimensions}, file_type="csv"
        )
