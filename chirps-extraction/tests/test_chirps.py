import os
import re
import tempfile
from datetime import date

import chirps
import geopandas as gpd
import pytest
import rasterio
import responses
from click.testing import CliRunner
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem


def test_filesystem():
    assert isinstance(chirps.filesystem("/tmp/file.txt"), LocalFileSystem)
    assert isinstance(chirps.filesystem("http://example.com/"), HTTPFileSystem)
    assert isinstance(chirps.filesystem("s3://bucket/dir"), S3FileSystem)
    # assert isinstance(chirps.filesystem("gcs://bucket/dir"), GCSFileSystem)
    with pytest.raises(ValueError):
        chirps.filesystem("bad://bucket/dir")


@pytest.fixture
def bfa_raw_data(boto_client):
    boto_client.create_bucket(Bucket="bfa-raw-data")
    for fname in os.listdir(
        os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017")
    ):
        with open(
            os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017", fname), "rb"
        ) as f:
            boto_client.put_object(
                Bucket="bfa-raw-data", Key=f"2017/{fname}", Body=f.read()
            )


@pytest.fixture(params=["s3", "file"])
def download_location(boto_client, request):
    if request.param == "s3":
        dirname = "s3://chirps-download"
        boto_client.create_bucket(Bucket="chirps-download")
        storage_options = {
            "client_kwargs": {"endpoint_url": os.environ["AWS_S3_ENDPOINT"]}
        }
    elif request.param == "file":
        dirname = os.path.join(os.path.dirname(__file__), "chirps-download")
        storage_options = {}
    else:
        raise NotImplementedError

    return request.param, dirname, storage_options


@pytest.fixture
def bfa_output_data(boto_client):
    boto_client.create_bucket(Bucket="bfa-output-data")


@pytest.fixture(scope="session")
def mock_chc():
    with responses.RequestsMock() as mocked_responses:

        def get_callback(request):
            file_name = (
                "chirps-v2.0.2017.05.05.tif.gz"
                if request.url.endswith(".gz")
                else "chirps-v2.0.2017.05.05.tif"
            )
            with open(
                os.path.join(os.path.dirname(__file__), "sample_tifs", file_name), "rb"
            ) as f:
                return 200, {"Content-Type": "application/x-gzip"}, f.read()

        def head_callback(request):
            return 200, {"Content-Type": "application/x-gzip"}, b""

        mocked_responses.add_callback(
            responses.GET,
            re.compile("https://data.chc.ucsb.edu/(.*)"),
            callback=get_callback,
        )

        mocked_responses.add_callback(
            responses.HEAD,
            re.compile("https://data.chc.ucsb.edu/(.*)"),
            callback=head_callback,
        )

        yield


def test__compress():
    src_raster = os.path.join(
        os.path.dirname(__file__), "sample_tifs", "chirps-v2.0.2017.05.05.tif"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        dst_raster = os.path.join(tmp_dir, "raster.tif")
        chirps._compress(src_raster, dst_raster)
        with rasterio.open(dst_raster) as src:
            assert src.profile.get("compress").lower() == "deflate"


def test__download(mock_chc):
    url = (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/"
        "2017/chirps-v2.0.2017.05.05.tif.gz"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        dst_file = os.path.join(tmp_dir, "chirps.tif")
        dst_file = chirps._download(url, dst_file)
        assert os.path.isfile(dst_file)


@pytest.fixture
def catalog(mock_chc):
    return chirps.Chirps(version="2.0", zone="africa", timely="daily")


def test_chirps_base_url(catalog):
    assert (
        catalog.base_url
        == "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05"
    )


def test_chirps_fname(catalog):
    assert catalog.fname(date(2020, 1, 1)) == "chirps-v2.0.2020.01.01.tif.gz"
    # no .gz after 2020-06-01
    assert catalog.fname(date(2021, 7, 1)) == "chirps-v2.0.2021.07.01.tif"


def test_chirps_download(catalog, mock_chc):

    with tempfile.TemporaryDirectory() as tmp_dir:

        dst_file = catalog.download(day=date(2017, 5, 5), output_dir=tmp_dir)

        with rasterio.open(dst_file) as src:
            assert src.profile.get("width")


def test_chirps_download_range(catalog, mock_chc):

    with tempfile.TemporaryDirectory() as tmp_dir:

        catalog.download_range(
            start=date(2017, 4, 15), end=date(2017, 5, 15), output_dir=tmp_dir
        )


def test_chirps_path():

    data_dir = os.path.join(os.path.dirname(__file__), "bfa-raw-data")
    expected = os.path.join(data_dir, "2017", "chirps-v2.0.2017.05.05.tif")
    assert chirps.chirps_path(data_dir, date(2017, 5, 5)) == expected


def test_raster_cumsum():

    bfa = gpd.read_file(os.path.join(os.path.dirname(__file__), "bfa.geojson"))
    extent = bfa.iloc[0].geometry

    data_dir = os.path.join(os.path.dirname(__file__), "bfa-raw-data")
    rasters = [
        os.path.join(data_dir, "2017", f)
        for f in [
            "chirps-v2.0.2017.05.01.tif",
            "chirps-v2.0.2017.05.02.tif",
            "chirps-v2.0.2017.05.03.tif",
            "chirps-v2.0.2017.05.04.tif",
        ]
    ]

    cumsum, affine, nodata = chirps.raster_cumsum(rasters, extent)
    assert cumsum.min() >= 0
    assert cumsum.max() <= 100
    assert cumsum.mean() == pytest.approx(15.85, abs=1)
    assert affine


def test_weekly_stats():

    contours = gpd.read_file(os.path.join(os.path.dirname(__file__), "bfa.geojson"))
    data_dir = os.path.join(os.path.dirname(__file__), "bfa-raw-data")
    start = date(2017, 4, 1)
    end = date(2017, 7, 1)

    stats = chirps.weekly_stats(contours, start, end, chirps_dir=data_dir)
    assert len(stats) > 50
    # todo: better quality checks


def test_monthly_stats():

    contours = gpd.read_file(os.path.join(os.path.dirname(__file__), "bfa.geojson"))
    data_dir = os.path.join(os.path.dirname(__file__), "bfa-raw-data")
    start = date(2017, 4, 1)
    end = date(2017, 7, 1)

    stats = chirps.monthly_stats(contours, start, end, chirps_dir=data_dir)
    assert len(stats) > 10
    # todo: better quality checks


def test__iter_month_days():

    assert len([d for d in chirps._iter_month_days(2020, 2)]) == 29
    assert len([d for d in chirps._iter_month_days(2021, 2)]) == 28


def test_epi_week():

    epiw = chirps.EpiWeek(date(2020, 1, 1))
    assert epiw.start == date(2019, 12, 29)
    assert epiw.end == date(2020, 1, 4)
    assert epiw.year == 2020
    assert epiw.week == 1

    assert chirps.EpiWeek(date(2020, 1, 2)) == chirps.EpiWeek(date(2019, 12, 30))


def test_epiweek_range():

    start = date(2020, 1, 1)
    end = date(2020, 3, 15)

    wrange = chirps.epiweek_range(start, end)
    assert len(wrange) == 12


def test_cli_download(moto_server, bfa_output_data, mock_chc):

    output_dir = os.path.join("s3://bfa-output-data/africa/daily")

    runner = CliRunner()
    result = runner.invoke(
        chirps.cli,
        [
            "download",
            "--output-dir",
            output_dir,
            "--start",
            "2017-04-30",
            "--end",
            "2017-06-01",
        ],
    )

    assert result.exit_code == 0


def test_cli_extract(moto_server, bfa_raw_data, bfa_output_data):

    chirps_dir = "s3://bfa-raw-data"
    contours = os.path.join(os.path.dirname(__file__), "bfa.geojson")
    weekly_output = "s3://bfa-output-data/weekly.csv"
    monthly_output = "s3://bfa-output-data/monthly.csv"

    runner = CliRunner()
    result = runner.invoke(
        chirps.cli,
        [
            "extract",
            "--input-dir",
            chirps_dir,
            "--contours",
            contours,
            "--start",
            "2017-04-30",
            "--end",
            "2017-06-01",
            "--weekly-csv",
            weekly_output,
            "--monthly-csv",
            monthly_output,
        ],
    )

    assert result.exit_code == 0
