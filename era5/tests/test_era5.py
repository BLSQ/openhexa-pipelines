import os
import shutil
import tempfile
from datetime import datetime

import geopandas as gpd
import numpy as np
from click.testing import CliRunner

import era5


def test_safe_from_injection():

    assert era5._safe_from_injection("weekly_auto") == "weekly_auto"
    assert era5._safe_from_injection("weekly_auto网络") == "weekly_auto"


def test_max_date_csv():

    fname = os.path.join(os.path.dirname(__file__), "data", "precipitation.csv")
    assert era5._max_date_csv(fname) == datetime(2017, 3, 31, 0, 0)


def test_human_readable_size():

    assert era5.human_readable_size(2140) == "2.1 KB"


def test_init_cdsapi():

    BOUNDS = (-5.6, 9.4, 2.5, 15.1)  # burkina faso
    api = era5.Era5(variable="total_precipitation", bounds=BOUNDS)
    api.init_cdsapi(api_key="0000", api_uid="0000")


def test_download(mocker):

    BOUNDS = (-5.6, 9.4, 2.5, 15.1)  # burkina faso

    with tempfile.TemporaryDirectory() as tmp_dir:

        dst_file = os.path.join(tmp_dir, "2m_temperature.nc")
        cache_dir = os.path.join(tmp_dir, "cache")
        os.makedirs(cache_dir)

        api = era5.Era5(
            variable="2m_temperature",
            bounds=BOUNDS,
            cache_dir=cache_dir,
        )
        api.init_cdsapi(api_key="", api_uid="")

        # mock the retrieve method of cdsapi to copy local file instead of
        # downloading a new file
        def mock_retrieve(*args, **kwargs):
            shutil.copyfile(
                os.path.join(os.path.dirname(__file__), "data", "2m_temperature.nc"),
                dst_file,
            )
            return

        mocker.patch("era5.cdsapi.Client.retrieve", mock_retrieve)

        api.download(year=2020, month=1, hours=["12:00"], dst_file=dst_file)
        assert os.path.isfile(dst_file)
        assert os.path.getsize(dst_file) > 0
        assert os.path.isfile(os.path.join(tmp_dir, "cache", "2m_temperature.nc"))

        # should hit cache
        # todo: find a way to verify that cache has indeed been used
        api.download(year=2020, month=1, hours=["12:00"], dst_file=dst_file)


def test_zonal_stats():

    tests_datadir = os.path.join(os.path.dirname(__file__), "data")

    stats = era5.zonal_stats(
        boundaries=gpd.read_file(
            os.path.join(tests_datadir, "bfa_gadm_level1.geojson")
        ),
        datafile=os.path.join(tests_datadir, "2m_temperature.nc"),
        agg_function=np.mean,
        variable_name=None,
    )

    assert stats.value.sum() > 0
    assert stats.period.min() == "2020-01-01"
    assert stats.period.max() == "2020-01-31"


def test_fix_geometries():

    bfa = gpd.read_file(
        os.path.join(os.path.dirname(__file__), "data", "bfa_gadm_level1.geojson")
    )
    era5.fix_geometries(bfa)


def test_cli_download(mocker):

    tests_datadir = os.path.join(os.path.dirname(__file__), "data")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:

        download_dir = os.path.join(tmp_dir, "download")
        os.makedirs(download_dir)

        # mock the retrieve method of cdsapi to copy local file instead of
        # downloading a new file
        def mock_retrieve(*args, **kwargs):
            request = args[2]
            fname = f"2m_temperature_{request['year']}{request['month']:02}.nc"
            shutil.copyfile(
                os.path.join(tests_datadir, fname),
                args[3],  # target parameter
            )

        mocker.patch("era5.cdsapi.Client.retrieve", mock_retrieve)

        result = runner.invoke(
            era5.download,
            [
                "--start",
                "2020-10-01",
                "--end",
                "2021-02-15",
                "--extent",
                os.path.join(tests_datadir, "bfa_gadm_level1.geojson"),
                "--cds-variable",
                "2m_temperature",
                "--hours",
                "06:00,12:00,18:00",
                "--output-dir",
                download_dir,
            ],
        )

        assert result.exit_code == 0
        assert "2m_temperature_202011.nc" in os.listdir(download_dir)


def test_cli_daily():

    tests_datadir = os.path.join(os.path.dirname(__file__), "data")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:

        input_dir = os.path.join(tmp_dir, "download")
        os.makedirs(input_dir)
        for fname in (
            "2m_temperature_202010.nc",
            "2m_temperature_202011.nc",
            "2m_temperature_202012.nc",
            "2m_temperature_202101.nc",
            "2m_temperature_202102.nc",
        ):
            shutil.copyfile(
                os.path.join(tests_datadir, fname), os.path.join(input_dir, fname)
            )

        result = runner.invoke(
            era5.daily,
            [
                "--input-dir",
                input_dir,
                "--boundaries",
                os.path.join(tests_datadir, "bfa_gadm_level1.geojson"),
                "--cds-variable",
                "2m_temperature",
                "--agg-function",
                "mean",
                "--csv",
                os.path.join(tmp_dir, "data.csv"),
            ],
        )

        assert result.exit_code == 0
        assert os.path.isfile(os.path.join(tmp_dir, "data.csv"))


def test_cli_weekly():

    tests_datadir = os.path.join(os.path.dirname(__file__), "data")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:

        result = runner.invoke(
            era5.aggregate,
            [
                "--src-file",
                os.path.join(tests_datadir, "2m_temperature.csv"),
                "--frequency",
                "weekly",
                "--agg-function",
                "mean",
                "--csv",
                os.path.join(tmp_dir, "data.csv"),
            ],
        )

        assert result.exit_code == 0
        assert os.path.isfile(os.path.join(tmp_dir, "data.csv"))


def test_cli_monthly():

    tests_datadir = os.path.join(os.path.dirname(__file__), "data")

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmp_dir:

        result = runner.invoke(
            era5.aggregate,
            [
                "--src-file",
                os.path.join(tests_datadir, "2m_temperature.csv"),
                "--frequency",
                "monthly",
                "--agg-function",
                "mean",
                "--csv",
                os.path.join(tmp_dir, "data.csv"),
            ],
        )

        assert result.exit_code == 0
        assert os.path.isfile(os.path.join(tmp_dir, "data.csv"))
