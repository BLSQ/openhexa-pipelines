import os
import pytest
import subprocess
import time
import botocore
import requests
import tempfile

from s3fs import S3FileSystem
from gcsfs import GCSFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem

import storage


S3_ENDPOINT_URL = "http://127.0.0.1:3000"


def test__no_protocol():
    assert storage._no_protocol("s3://bucket/dir/file.txt") == "bucket/dir/file.txt"
    assert storage._no_protocol("/tmp/file.txt") == "/tmp/file.txt"


def test_filesystem():
    assert isinstance(storage.filesystem("/tmp/file.txt"), LocalFileSystem)
    assert isinstance(storage.filesystem("http://example.com/"), HTTPFileSystem)
    assert isinstance(storage.filesystem("s3://bucket/dir"), S3FileSystem)
    assert isinstance(storage.filesystem("gcs://bucket/dir"), GCSFileSystem)
    with pytest.raises(ValueError):
        storage.filesystem("bad://bucket/dir")


@pytest.fixture()
def moto_server():

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_S3_ENDPOINT" not in os.environ:
        os.environ["AWS_S3_ENDPOINT"] = S3_ENDPOINT_URL

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


def test_open(moto_server):

    # Local
    with tempfile.NamedTemporaryFile() as tmp_file:
        with storage.open(tmp_file.name, "w") as f:
            f.write("hello world")
        with storage.open(tmp_file.name) as f:
            assert f.read() == b"hello world"

    # S3
    moto_create_bucket("test-bucket")
    path = "s3://test-bucket/file.txt"
    with storage.open(path, "w") as f:
        f.write("hello world")
    with storage.open(path) as f:
        assert f.read() == b"hello world"


def test_exists(moto_server):

    # Local
    assert storage.exists("tests/bfa.geojson")

    # S3
    moto_put_test_data("test-bucket")
    assert storage.exists("s3://test-bucket/2017-18/chirps-v2.0.2017.04.30.tif")


def test_glob(moto_server):

    # Local
    result = storage.glob("tests/bfa-raw-data/2017-*/chirps*.tif")
    assert len(result) == 7
    assert result[0].startswith("/")  # path should be absolute
    assert result[0].endswith("tests/bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif")

    # S3
    moto_put_test_data("test-bucket")
    result = storage.glob("s3://test-bucket/2017-*/chirps*.tif")
    assert len(result) == 7
    assert result[0] == "s3://test-bucket/2017-18/chirps-v2.0.2017.04.30.tif"


def test_makedirs():

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = os.path.join(tmp_dir, "test1", "test2")
        storage.makedirs(test_dir)
        assert os.path.isdir(test_dir)
        storage.makedirs(test_dir)  # Directory already exists


def test_size(moto_server):

    # Local
    assert (
        storage.size("tests/bfa-raw-data/2017-18/chirps-v2.0.2017.04.30.tif") == 72930
    )

    # S3
    moto_put_test_data("test-bucket")
    assert storage.size("s3://test-bucket/2017-18/chirps-v2.0.2017.04.30.tif") == 72930


def test_put(moto_server):

    moto_create_bucket("test-bucket")
    storage.put("tests/bfa.geojson", "s3://test-bucket/dir/bfa.geojson")
    assert storage.exists("s3://test-bucket/dir/bfa.geojson")


def test_get(moto_server):

    moto_put_test_data("test-bucket")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dst_file = os.path.join(tmp_dir, "raster.tif")
        storage.get("s3://test-bucket/2017-18/chirps-v2.0.2017.04.30.tif", dst_file)
        assert os.path.isfile(dst_file)
