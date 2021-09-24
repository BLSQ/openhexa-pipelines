import os
import re
import subprocess
import time
from datetime import date

import botocore
import pandas as pd
import responses
from s3fs import S3FileSystem

import chirps
import pytest
import rasterio
import requests


@pytest.fixture()
def moto_server():
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_S3_ENDPOINT" not in os.environ:
        os.environ["AWS_S3_ENDPOINT"] = "http://127.0.0.1:3000"

    p = subprocess.Popen(["moto_server", "s3", "-p", "3000"])

    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(os.environ["AWS_S3_ENDPOINT"])
            if r.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    yield
    p.terminate()
    p.wait()


def moto_put_test_data(bucket_name):
    # put raw BFA data into a test bucket
    session = botocore.session.Session()
    client = session.create_client("s3", endpoint_url=os.environ["AWS_S3_ENDPOINT"])
    client.create_bucket(Bucket=bucket_name)
    for fname in os.listdir(
        os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017-18")
    ):
        with open(
            os.path.join(os.path.dirname(__file__), "bfa-raw-data/2017-18", fname), "rb"
        ) as f:
            client.put_object(Bucket=bucket_name, Key=f"2017-18/{fname}", Body=f.read())


def moto_create_bucket(bucket_name):
    # create an empty writable bucket
    session = botocore.session.Session()
    client = session.create_client("s3", endpoint_url=os.environ["AWS_S3_ENDPOINT"])
    client.create_bucket(Bucket=bucket_name)


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


def test_download_chirps_daily(moto_server, mocked_responses):
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

    moto_create_bucket("test-bucket")
    output_dir = "s3://test-bucket/africa/daily"
    os.environ["AWS_S3_ENDPOINT"] = "http://localhost:3000"
    chirps.download_chirps_daily(output_dir=output_dir, year_start=2012, year_end=2012)

    fs = S3FileSystem(client_kwargs={"endpoint_url": os.environ["AWS_S3_ENDPOINT"]})
    dirs = fs.ls(f"{output_dir}")
    assert len(dirs) == 53
    files = fs.glob(f"{output_dir}/*/*.tif")
    assert len(files) == 366

    # TODO: finish
    # with rasterio.open(f"s3://{files[0]}") as src:
    #     foo = src.profile
    #     bar = "baz"


def test__no_ending_slash():
    assert chirps._no_ending_slash("bucket/folder/") == "bucket/folder"
    assert chirps._no_ending_slash("bucket/folder") == "bucket/folder"


def test_extract_sum_epi_week(moto_server):
    moto_put_test_data("bfa-raw-data")

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
        sum, affine = chirps.extract_sum_epi_week(files)
    assert sum.min() >= 0
    assert isinstance(affine, rasterio.Affine)


def test_extract_chirps_data(moto_server):
    moto_put_test_data("bfa-raw-data")
    moto_create_bucket("bfa-output-data")

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
