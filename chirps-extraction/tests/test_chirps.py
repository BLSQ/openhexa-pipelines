import os
import subprocess
import time
from datetime import date

import botocore
import chirps
import geopandas as gpd
import pytest
import rasterio
import requests
import s3fs


S3_ENDPOINT_URL = "http://127.0.0.1:3000"


@pytest.fixture()
def moto_server():

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"

    p = subprocess.Popen(["moto_server", "s3", "-p", "3000"])

    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(S3_ENDPOINT_URL)
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
    client = session.create_client("s3", endpoint_url=S3_ENDPOINT_URL)
    client.create_bucket(Bucket=bucket_name)
    for fname in os.listdir("tests/bfa-raw-data/2017-18"):
        with open(os.path.join("tests/bfa-raw-data/2017-18", fname), "rb") as f:
            client.put_object(Bucket=bucket_name, Key=f"2017-18/{fname}", Body=f.read())


def moto_create_bucket(bucket_name):
    # create an empty writable bucket
    session = botocore.session.Session()
    client = session.create_client("s3", endpoint_url=S3_ENDPOINT_URL)
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


def test_download_chirps_data(moto_server, monkeypatch):

    # return empty bytes instead of the actual data
    def mockreturn(self, chunk_size):
        return [b"", b"", b""]

    monkeypatch.setattr(requests.Response, "iter_content", mockreturn)

    moto_create_bucket("test-bucket")
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": S3_ENDPOINT_URL})
    url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/2021/chirps-v2.0.2021.06.06.tif"
    output_file = "s3://test-bucket/subfolder/raster.tif"
    chirps.download_chirps_data(fs, url, output_file)
    assert fs.exists(output_file)


def test_download_chirps_daily(moto_server, monkeypatch):
    def mockreturn(fs, url, output_path):
        return ""

    monkeypatch.setattr(chirps, "download_chirps_data", mockreturn)

    moto_create_bucket("test-bucket")
    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": S3_ENDPOINT_URL})
    output_dir = "s3://test-bucket/africa/daily"
    chirps.download_chirps_daily(fs, output_dir, 2012, 2013)


def test__no_ending_slash():

    assert chirps._no_ending_slash("bucket/folder/") == "bucket/folder"
    assert chirps._no_ending_slash("bucket/folder") == "bucket/folder"


def test_rio_read_file(moto_server):

    moto_put_test_data("bfa-raw-data")
    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=S3_ENDPOINT_URL.replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        data = chirps.rio_read_file(
            "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif"
        )
    assert data.min() >= 0


def test_rio_get_affine(moto_server):

    moto_put_test_data("bfa-raw-data")
    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=S3_ENDPOINT_URL.replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        affine = chirps.rio_get_affine(
            "s3://bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif"
        )
    assert isinstance(affine, rasterio.Affine)


def test_extract_sum_epi_week(moto_server):

    moto_put_test_data("bfa-raw-data")

    files = [
        "bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.01.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.02.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.03.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.04.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.05.tif",
        "bfa-raw-data/2017-18/chirps-v2.0.2017.05.06.tif",
    ]

    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=S3_ENDPOINT_URL.replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        sum, affine = chirps.extract_sum_epi_week(files)
    assert sum.min() >= 0
    assert isinstance(affine, rasterio.Affine)


def test_extract_chirps_data(moto_server):

    moto_put_test_data("bfa-raw-data")

    fs = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": S3_ENDPOINT_URL})

    contours = gpd.read_file("tests/bfa.geojson")

    with rasterio.Env(
        AWS_HTTPS=False,
        AWS_VIRTUAL_HOSTING=False,
        AWS_NO_SIGN_REQUEST=False,
        AWS_S3_ENDPOINT=S3_ENDPOINT_URL.replace("http://", ""),
        AWS_DEFAULT_REGION="us-east-1",
    ):
        stats = chirps.extract_chirps_data(
            fs=fs,
            contours=contours,
            input_dir="s3://bfa-raw-data",
            year_start=2017,
            year_end=2018,
        )

    assert len(stats) == 13
