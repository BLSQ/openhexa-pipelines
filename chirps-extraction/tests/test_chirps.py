import os
import re
from datetime import date

import botocore
import fsspec
import pandas as pd
import pytest
import responses

import chirps
import rasterio


@pytest.fixture
def boto_client():
    session = botocore.session.Session()

    return session.create_client("s3", endpoint_url=os.environ["AWS_S3_ENDPOINT"])


@pytest.fixture
def bfa_raw_data(boto_client):
    boto_client.create_bucket(Bucket="bfa-raw-data")
    for fname in os.listdir(
        os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017-18")
    ):
        with open(
            os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017-18", fname), "rb"
        ) as f:
            boto_client.put_object(
                Bucket="bfa-raw-data", Key=f"2017-18/{fname}", Body=f.read()
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


def test_provide_epi_week():
    # epi_year == year
    epi_week, epi_year, start, end = chirps.provide_epi_week(2012, 1, 12)
    assert epi_week == 2
    assert epi_year == 2012
    assert start == date(2012, 1, 8)
    assert end == date(2012, 1, 14)

    # epi_year != year
    epi_week, epi_year, start, end = chirps.provide_epi_week(2018, 12, 30)
    assert epi_week == 1
    assert epi_year == 2019
    assert start == date(2018, 12, 30)
    assert end == date(2019, 1, 5)


def test_provide_time_range():
    drange = chirps.provide_time_range(2012, 2014)
    assert len(drange) == 1096
    assert drange[0].date() == date(2012, 1, 1)
    assert drange[-1].date() == date(2014, 12, 31)


@pytest.fixture
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


def test_download_chirps_daily(moto_server, mock_chc, download_location):
    protocol, download_dir, storage_options = download_location
    output_dir = os.path.join(download_dir, "africa/daily")
    chirps.download_chirps_daily(
        output_dir=output_dir, year_start=2012, year_end=2012, overwrite=True
    )

    fs = fsspec.filesystem(protocol, **storage_options)
    dirs = fs.ls(f"{output_dir}")
    assert len(dirs) == 53
    files = fs.glob(f"{output_dir}/*/*.tif")
    assert len(files) == 366


def test_extract_sum_epi_week(moto_server, bfa_raw_data):
    files = [
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.01.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.02.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.03.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.04.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.05.tif",
        "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.05.06.tif",
    ]

    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=os.environ["AWS_S3_ENDPOINT"].replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        fs = fsspec.filesystem(
            "s3", client_kwargs={"endpoint_url": os.environ["AWS_S3_ENDPOINT"]}
        )
        prec_sum, affine = chirps.extract_sum_epi_week(fs, files)
    assert prec_sum.min() >= 0
    assert isinstance(affine, rasterio.Affine)


def test_extract_chirps_data(moto_server, bfa_raw_data, bfa_output_data):
    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=os.environ["AWS_S3_ENDPOINT"].replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        chirps.extract_chirps_data(
            contours_file=os.path.join(os.path.dirname(__file__), "bfa.geojson"),
            input_dir="s3://bfa-raw-data",
            output_file="s3://bfa-output-data/results.csv",
            start_year=2017,
            end_year=2018,
        )

    output_df = pd.read_csv(
        "s3://bfa-output-data/results.csv",
        storage_options={
            "client_kwargs": {"endpoint_url": os.environ["AWS_S3_ENDPOINT"]}
        },
    )
    assert isinstance(output_df, pd.DataFrame)
    assert len(output_df) == 13
