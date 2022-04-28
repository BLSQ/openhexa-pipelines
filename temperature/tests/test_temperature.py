import os
import re
import tempfile

import pytest
import responses
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

import temperature


def test_filesystem():
    assert isinstance(temperature.filesystem("/tmp/file.txt"), LocalFileSystem)
    assert isinstance(temperature.filesystem("s3://bucket/dir"), S3FileSystem)
    with pytest.raises(ValueError):
        temperature.filesystem("bad://bucket/dir")


@responses.activate
def test_sync_data():

    with tempfile.TemporaryDirectory() as tmp_dir:

        responses.add(
            responses.HEAD,
            url=re.compile("https://downloads.psl.noaa.gov.*nc"),
            status=200,
            headers={"content-length": "1"},
        )

        responses.add(
            responses.GET,
            url=re.compile("https://downloads.psl.noaa.gov.*nc"),
            body=b"0",
            status=200,
        )

        for var in ("tmin", "tmax"):
            temperature.sync_data(
                data_dir=tmp_dir, start_year=2015, end_year=2020, variable=var
            )

        files = os.listdir(tmp_dir)
        for year in range(2015, 2020 + 1):
            for var in ("tmin", "tmax"):
                assert os.path.exists(os.path.join(tmp_dir, f"{var}.{year}.nc"))
