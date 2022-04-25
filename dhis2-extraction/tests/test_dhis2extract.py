import json
import os
import re
import tempfile
from io import StringIO

import dhis2extract
import pandas as pd
import pytest
import responses
from click.testing import CliRunner
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

DHIS_INSTANCE = "play.dhis2.org/2.34.7"


def test_filesystem():
    assert isinstance(dhis2extract.filesystem("/tmp/file.txt"), LocalFileSystem)
    assert isinstance(dhis2extract.filesystem("http://example.com/"), HTTPFileSystem)
    assert isinstance(dhis2extract.filesystem("s3://bucket/dir"), S3FileSystem)
    # assert isinstance(dhis2extract.filesystem("gcs://bucket/dir"), GCSFileSystem)
    with pytest.raises(ValueError):
        dhis2extract.filesystem("bad://bucket/dir")


@pytest.fixture(scope="module")
@responses.activate
def demo():
    """DHIS2 demo instance APIs."""

    # mock api/metadata calls before creating the DHIS2 object
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    for fname in os.listdir(os.path.join(responses_dir, "metadata")):
        name = fname.split(".")[0]
        with open(os.path.join(responses_dir, "metadata", fname)) as f:
            responses.add(
                responses.GET,
                url=re.compile(f".+metadata.+{name}=True.+"),
                body=f.read(),
                status=200,
            )

    return dhis2extract.DHIS2(DHIS_INSTANCE, "", "")


def test_dhis2_get_metadata(demo):
    assert len(demo.metadata) == 12
    assert len(demo.metadata.get("organisationUnits")) == 7860


def test_data_element_dataset(demo):
    assert demo.data_element_dataset("l6byfWFUGaP") == "BfMAe6Itzgt"
    assert demo.data_element_dataset("Bad UID") is None


@pytest.mark.parametrize("level,expected", [(0, 0), (3, 267), (4, 2897)])
def test_org_units_per_lvl(demo, level, expected):
    assert len(demo.org_units_per_lvl(level)) == expected


@responses.activate
def test_data_value_sets_01(demo):
    """With start and end dates arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "dataValueSets", "response01.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/dataValueSets.csv.+startDate=.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.data_value_sets(
        start_date="2020-01-01",
        end_date="2020-03-01",
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 302


@responses.activate
def test_data_value_sets_02(demo):
    """With periods arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "dataValueSets", "response02.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/dataValueSets.csv.+period=202004.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.data_value_sets(
        periods=["202004", "202006"],
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 325


@responses.activate
def test_data_value_sets_03(demo):
    """With datasets arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "dataValueSets", "response03.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/dataValueSets.csv.+period=202008.+orgUnit=VdXuxcNkiad"),
            body=f.read(),
            status=200,
        )
    csv = demo.data_value_sets(
        periods=["202008", "202010"],
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        datasets=["BfMAe6Itzgt", "QX4ZTUbOt3a"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 196


@responses.activate
def test_data_value_sets_04(demo, raw_metadata):
    """With loads of org units - requests should be chunked"""

    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "dataValueSets", "response04.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/dataValueSets.csv.+period=202008.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.data_value_sets(
        periods=["202008", "202010"],
        org_units=[ou["id"] for ou in raw_metadata["organisationUnits"][:113]],
        datasets=["BfMAe6Itzgt", "QX4ZTUbOt3a"],
    )
    df = pd.read_csv(StringIO(csv))

    # 113 org units, chunk size=50, we should have 3 requests (they share the same response with 196 lines)
    assert len(df) == 196 * 3


@responses.activate
def test_data_value_sets_05(demo):
    """With levels arguments - requests should be chunked."""

    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "dataValueSets", "response04.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/dataValueSets.csv.+children=True.+period=202008.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.data_value_sets(
        periods=["202008", "202010"],
        org_unit_levels=[4],
        datasets=["BfMAe6Itzgt", "QX4ZTUbOt3a"],
    )
    df = pd.read_csv(StringIO(csv))

    # 267 org units for level 4, chunk size=50, we should have 6 requests (they share the same response with 196 lines)
    assert len(df) == 196 * 6


@responses.activate
def test_analytics_01(demo):
    """With start and end dates arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "analytics", "response01.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/analytics.csv.+pe%3A202001%3B202002%3B202003.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.analytics(
        start_date="202001",
        end_date="202003",
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 25


@responses.activate
def test_analytics_02(demo):
    """With periods arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "analytics", "response02.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/analytics.csv.+pe%3A202004%3B202006.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.analytics(
        periods=["202004", "202006"],
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 21


@responses.activate
def test_analytics_raw_data_01(demo):
    """With start and end dates arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "analyticsRawData", "response01.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/analytics/rawData.csv.+startDate=2020-01-01.+"),
            body=f.read(),
            status=200,
        )
    csv = demo.analytics_raw_data(
        start_date="2020-01-01",
        end_date="2020-03-01",
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 17


@responses.activate
def test_analytics_raw_data_02(demo):
    """With periods arguments."""
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    with open(os.path.join(responses_dir, "analyticsRawData", "response02.csv")) as f:
        responses.add(
            responses.GET,
            url=re.compile(".+/analytics/rawData.csv.+pe%3A202004%3B202006.*"),
            body=f.read(),
            status=200,
        )
    csv = demo.analytics_raw_data(
        periods=["202004", "202006"],
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
    )
    df = pd.read_csv(StringIO(csv))
    assert len(df) == 32


def test_dimension_param():

    param = dhis2extract._dimension_param(
        org_units=["VdXuxcNkiad", "BNFrspDBKel"],
        org_unit_groups=["tDZVQ1WtwpA", "MAs88nJc9nL"],
        org_unit_levels=[4, 5],
        data_elements=["l6byfWFUGaP", "Boy3QwztgeZ"],
        indicators=["Uvn6LCg7dVU", "aGByu8NFs9m"],
        programs=["bMcwwoVnbSR"],
        periods=["202001", "202002", "202004"],
    )

    # periods
    assert "pe:202001;202002;202004" in param
    # org units
    assert "ou:VdXuxcNkiad;BNFrspDBKel" in param
    # org unit groups
    assert "ou:OU_GROUP-tDZVQ1WtwpA;OU_GROUP-MAs88nJc9nL" in param
    # org unit levels
    assert "ou:LEVEL-4;LEVEL-5" in param
    # data elements
    assert "dx:l6byfWFUGaP;Boy3QwztgeZ" in param
    # indicators
    assert "dx:Uvn6LCg7dVU;aGByu8NFs9m" in param
    # programs
    assert "dx:bMcwwoVnbSR" in param


def test_check_iso_date():

    assert not dhis2extract._check_iso_date("202001")
    assert dhis2extract._check_iso_date("2020-01-01")


def test_check_dhis2_period():

    assert not dhis2extract._check_dhis2_period("2020-01-01")
    assert dhis2extract._check_dhis2_period("2020Q1")
    assert dhis2extract._check_dhis2_period("2020")
    assert dhis2extract._check_dhis2_period("202001")


@pytest.fixture(scope="module")
def raw_metadata():
    fpath = os.path.join(os.path.dirname(__file__), "data", "metadata.json")
    with open(fpath) as f:
        return json.load(f)


def test_transform_org_units(raw_metadata):
    df = dhis2extract._transform_org_units(raw_metadata)
    assert len(df) > 10
    assert "Rp268JB6Ne4" in df.ou_uid.unique()
    for column in ["ou_uid", "ou_name", "path"]:
        assert column in df.columns


def test_transform_org_units_geo(raw_metadata):
    fpath = os.path.join(os.path.dirname(__file__), "data", "org_units.csv")
    df = pd.read_csv(fpath)
    geodf = dhis2extract._transform_org_units_geo(df)
    assert len(geodf) > 10
    assert "Rp268JB6Ne4" in df.ou_uid.unique()
    geodf = geodf[pd.notna(geodf.geometry)]
    assert len(geodf) > 10


def test_transform_org_unit_groups(raw_metadata):
    df = dhis2extract._transform_org_unit_groups(raw_metadata)
    assert len(df) > 10
    assert "CXw2yu5fodb" in df.oug_uid.unique()
    for column in ["oug_uid", "oug_name"]:
        assert column in df.columns


def test_transform_data_elements(raw_metadata):
    df = dhis2extract._transform_data_elements(raw_metadata)
    assert len(df) > 10
    assert "FTRrcoaog83" in df.dx_uid.unique()
    for column in ["dx_uid", "dx_name"]:
        assert column in df.columns


def test_transform_data_element_groups(raw_metadata):
    df = dhis2extract._transform_data_element_groups(raw_metadata)
    assert len(df) > 1
    for column in ["deg_uid", "deg_name"]:
        assert column in df.columns


def test_transform_indicators(raw_metadata):
    df = dhis2extract._transform_indicators(raw_metadata)
    assert len(df) > 10
    assert "ReUHfIn0pTQ" in df.dx_uid.unique()
    for column in ["dx_uid", "dx_name"]:
        assert column in df.columns


def test_transform_indicator_groups(raw_metadata):
    df = dhis2extract._transform_indicator_groups(raw_metadata)
    assert len(df) > 10
    assert "oehv9EO3vP7" in df.ing_uid.unique()
    for column in ["ing_uid", "ing_name"]:
        assert column in df.columns


def test_transform_datasets(raw_metadata):
    df = dhis2extract._transform_datasets(raw_metadata)
    assert len(df) > 10
    assert "lyLU2wR22tC" in df.ds_uid.unique()
    for column in ["ds_uid", "ds_name", "data_elements", "org_units"]:
        assert column in df.columns


def test_transform_programs(raw_metadata):
    df = dhis2extract._transform_programs(raw_metadata)
    assert len(df) > 10
    assert "lxAQ7Zs9VYR" in df.program_uid.unique()
    for column in ["program_uid", "program_name"]:
        assert column in df.columns


def test_transform_cat_combos(raw_metadata):
    df = dhis2extract._transform_cat_combos(raw_metadata)
    assert len(df) > 10
    assert "m2jTvAj5kkm" in df.cc_uid.unique()
    for column in ["cc_uid", "cc_name"]:
        assert column in df.columns


def test_transform_cat_options(raw_metadata):
    df = dhis2extract._transform_cat_options(raw_metadata)
    assert len(df) > 10
    assert "FbLZS3ueWbQ" in df.co_uid.unique()
    for column in ["co_uid", "co_name"]:
        assert column in df.columns


def test_transform_categories(raw_metadata):
    df = dhis2extract._transform_categories(raw_metadata)
    assert len(df) > 10
    assert "KfdsGBcoiCa" in df.cat_uid.unique()
    for column in ["cat_uid", "cat_name"]:
        assert column in df.columns


def test_transform_category_option_combos(raw_metadata):
    df = dhis2extract._transform_category_option_combos(raw_metadata)
    assert len(df) > 10
    assert "S34ULMcHMca" in df.coc_uid.unique()
    for column in ["coc_uid", "coc_name"]:
        assert column in df.columns


def test_transform_analytics(raw_metadata):
    fpath = os.path.join(
        os.path.dirname(__file__), "responses", "analytics", "response01.csv"
    )
    df = dhis2extract._transform(pd.read_csv(fpath))
    assert len(df) == 25
    assert "Boy3QwztgeZ" in df.dx_uid.unique()
    assert "rQLFnNXXIL0" in df.coc_uid.unique()
    assert "VdXuxcNkiad" in df.ou_uid.unique()
    for column in ["dx_uid", "coc_uid", "period", "ou_uid", "value"]:
        assert column in df.columns


def test_transform_data_value_sets(raw_metadata):
    fpath = os.path.join(
        os.path.dirname(__file__), "responses", "dataValueSets", "response01.csv"
    )
    df = dhis2extract._transform(pd.read_csv(fpath))
    assert len(df) == 302
    assert "dY4OCwl0Y7Y" in df.dx_uid.unique()
    assert "J2Qf1jtZuj8" in df.coc_uid.unique()
    assert "VdXuxcNkiad" in df.ou_uid.unique()
    for column in ["dx_uid", "coc_uid", "period", "ou_uid", "value"]:
        assert column in df.columns


def test_transform_analytics_raw_data(raw_metadata):
    fpath = os.path.join(
        os.path.dirname(__file__), "responses", "analyticsRawData", "response01.csv"
    )
    df = dhis2extract._transform(pd.read_csv(fpath))
    assert len(df) == 17
    assert "l6byfWFUGaP" in df.dx_uid.unique()
    assert "Prlt0C1RF0s" in df.coc_uid.unique()
    assert "BNFrspDBKel" in df.ou_uid.unique()
    for column in ["dx_uid", "coc_uid", "period", "ou_uid", "value"]:
        assert column in df.columns


@pytest.fixture(scope="module")
def data_elements():
    """Data elements metadata."""
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "data_elements.csv"),
        index_col=0,
    )


@pytest.fixture(scope="module")
def indicators():
    """Indicators metadata."""
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "indicators.csv"), index_col=0
    )


@pytest.fixture(scope="module")
def category_option_combos():
    """COC metadata."""
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "category_option_combos.csv"),
        index_col=0,
    )


@pytest.fixture(scope="module")
def organisation_units():
    """Org units metadata."""
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "organisation_units.csv"),
        index_col=0,
    )


@pytest.mark.parametrize(
    "dx_uid,expected", [("ReUHfIn0pTQ", "Indicator"), ("FTRrcoaog83", "Data Element")]
)
def test_dx_type(dx_uid, expected, data_elements, indicators):
    assert dhis2extract._dx_type(dx_uid, data_elements, indicators) == expected


@pytest.mark.parametrize(
    "dx_uid,expected",
    [("A21lT9x7pmc", "Cabin fever"), ("aGByu8NFs9m", "Well nourished rate")],
)
def test_dx_name(dx_uid, expected, data_elements, indicators):
    assert dhis2extract._dx_name(dx_uid, data_elements, indicators) == expected


@pytest.mark.parametrize(
    "level,expected", [(1, "ImspTQPwCqd"), (3, "qtr8GGlm4gg"), (5, None)]
)
def test_level_uid(level, expected):
    assert (
        dhis2extract._level_uid(
            "/ImspTQPwCqd/at6UHUQatSo/qtr8GGlm4gg/Rp268JB6Ne4", level
        )
        == expected
    )


def test_join_metadata(
    data_elements, indicators, category_option_combos, organisation_units
):
    extract = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "extract.csv")
    )
    merge = dhis2extract._join_from_metadata(
        extract, data_elements, indicators, category_option_combos, organisation_units
    )

    for column in ["dx_name", "dx_type", "coc_name", "level_1_uid", "level_1_name"]:
        assert column in merge.columns

    assert len(merge) == len(extract)
    assert merge.isna().values.sum() == 0
    assert (merge.level_1_name == "Sierra Leone").all()


def test_add_empty_rows():
    extract = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "extract.csv")
    )
    df = dhis2extract._add_empty_rows(
        extract, index_columns=["dx_uid", "coc_uid", "period", "ou_uid"]
    )
    assert len(extract) == 25
    assert len(df) == 60
    assert len(df[pd.isna(df.value)]) >= 1


@pytest.mark.skip(reason="responses not mocked")
def test_download_data_value_sets():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = runner.invoke(
            dhis2extract.cli,
            [
                "download",
                "--instance",
                DHIS_INSTANCE,
                "--username",
                "",
                "--password",
                "",
                "--output-dir",
                tmp_dir,
                "--start",
                "2020-01-01",
                "--end",
                "2020-03-01",
                "-ou",
                "VdXuxcNkiad",
                "-ou",
                "BNFrspDBKel",
                "-de",
                "l6byfWFUGaP",
                "-de",
                "Boy3QwztgeZ",
                "--no-aggregate",
                "--no-analytics",
            ],
        )
        assert result.exit_code == 0
        assert "data_value_sets.csv" in os.listdir(tmp_dir)
        assert "metadata.json" in os.listdir(tmp_dir)


@pytest.mark.skip(reason="responses not mocked")
def test_download_analytics():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = runner.invoke(
            dhis2extract.cli,
            [
                "download",
                "--instance",
                DHIS_INSTANCE,
                "--username",
                "",
                "--password",
                "",
                "--output-dir",
                tmp_dir,
                "--start",
                "202001",
                "--end",
                "202003",
                "-ou",
                "VdXuxcNkiad",
                "-ou",
                "BNFrspDBKel",
                "-de",
                "l6byfWFUGaP",
                "-de",
                "Boy3QwztgeZ",
                "--aggregate",
                "--analytics",
            ],
        )
        assert result.exit_code == 0
        assert "analytics.csv" in os.listdir(tmp_dir)
        assert "metadata.json" in os.listdir(tmp_dir)


@pytest.mark.skip(reason="responses not mocked")
def test_download_analytics_raw_data():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = runner.invoke(
            dhis2extract.cli,
            [
                "download",
                "--instance",
                DHIS_INSTANCE,
                "--username",
                "",
                "--password",
                "",
                "--output-dir",
                tmp_dir,
                "--start",
                "2020-01-01",
                "--end",
                "2020-03-01",
                "-ou",
                "VdXuxcNkiad",
                "-ou",
                "BNFrspDBKel",
                "-de",
                "l6byfWFUGaP",
                "-de",
                "Boy3QwztgeZ",
                "--no-aggregate",
                "--analytics",
            ],
        )
        assert result.exit_code == 0
        assert "analytics_raw_data.csv" in os.listdir(tmp_dir)
        assert "metadata.json" in os.listdir(tmp_dir)


@pytest.mark.skip(reason="responses not mocked")
def test_download_transform():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = runner.invoke(
            dhis2extract.cli,
            [
                "download",
                "--instance",
                DHIS_INSTANCE,
                "--username",
                "",
                "--password",
                "",
                "--output-dir",
                tmp_dir,
                "--start",
                "202001",
                "--end",
                "202003",
                "-ou",
                "VdXuxcNkiad",
                "-ou",
                "BNFrspDBKel",
                "-de",
                "l6byfWFUGaP",
                "-de",
                "Boy3QwztgeZ",
                "--aggregate",
                "--analytics",
            ],
        )
        assert result.exit_code == 0

        result = runner.invoke(
            dhis2extract.cli, ["transform", "-i", tmp_dir, "-o", tmp_dir]
        )

        assert result.exit_code == 0
        assert "extract.csv" in os.listdir(tmp_dir)
        for fname in [
            "organisation_units.gpkg",
            "categories.csv",
            "category_combos.csv",
            "category_options.csv",
            "category_option_combos.csv",
            "programs.csv",
            "datasets.csv",
            "indicator_groups.csv",
            "indicators.csv",
            "data_elements.csv",
            "data_element_groups.csv",
            "organisation_unit_groups.csv",
            "organisation_units.csv",
        ]:
            assert fname in os.listdir(os.path.join(tmp_dir, "metadata"))
