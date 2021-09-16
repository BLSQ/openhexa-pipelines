import os
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import date

import chirps
import psutil
import pytest
import rasterio
import requests
import s3fs


@contextmanager
def minio_serve(data_dir):
    """A Context Manager to launch and close a MinIO server."""
    p = subprocess.Popen(["minio", "server", "--address", ":9001", data_dir])
    try:
        yield p
    finally:
        psutil.Process(p.pid).kill()


@pytest.fixture
def mock_s3fs(monkeypatch):
    """Mock storage.get_s3fs() to use local MinIO server."""

    def mockreturn():
        return s3fs.S3FileSystem(
            key="minioadmin",
            secret="minioadmin",
            client_kwargs={"endpoint_url": "http://localhost:9001", "region_name": ""},
        )

    monkeypatch.setattr(s3fs, "S3FileSystem", mockreturn)


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


def test_download_chirps_data(monkeypatch):

    # return empty bytes instead of the actual data
    def mockreturn(self, chunk_size):
        return [b"", b"", b""]

    monkeypatch.setattr(requests.Response, "iter_content", mockreturn)

    # launch a local MinIO server and setup s3fs accordingly
    with tempfile.TemporaryDirectory(prefix="chirps_") as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "bucket"))
        with minio_serve(tmp_dir):
            fs = s3fs.S3FileSystem(
                key="minioadmin",
                secret="minioadmin",
                client_kwargs={
                    "endpoint_url": "http://localhost:9001",
                    "region_name": "",
                },
            )

            url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/2021/chirps-v2.0.2021.06.06.tif"
            output_file = "s3://bucket/subfolder/raster.tif"
            chirps.download_chirps_data(fs, url, output_file)
            assert fs.exists(output_file)


def test_download_chirps_daily(monkeypatch):
    def mockreturn(fs, url, output_path):
        return ""

    monkeypatch.setattr(chirps, "download_chirps_data", mockreturn)

    # launch a local MinIO server and setup s3fs accordingly
    with tempfile.TemporaryDirectory(prefix="chirps_") as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "bucket"))
        with minio_serve(tmp_dir):
            fs = s3fs.S3FileSystem(
                key="minioadmin",
                secret="minioadmin",
                client_kwargs={
                    "endpoint_url": "http://localhost:9001",
                    "region_name": "",
                },
            )

            output_dir = "s3://bucket/africa/daily"
            chirps.download_chirps_daily(fs, output_dir, 2012, 2013)


def test_rio_get_affine():

    fname = os.path.join(
        "tests", "bfa-raw-data", "2017-18", "chirps-v2.0.2017.04.30.tif"
    )
    transform = chirps.rio_get_affine(fname)
    assert isinstance(transform, rasterio.Affine)


def test_rio_read_file():

    fname = os.path.join(
        "tests", "bfa-raw-data", "2017-18", "chirps-v2.0.2017.04.30.tif"
    )
    data = chirps.rio_read_file(fname)
    assert data.min() >= 0


def test__no_ending_slash():

    assert chirps._no_ending_slash("bucket/folder/") == "bucket/folder"
    assert chirps._no_ending_slash("bucket/folder") == "bucket/folder"
