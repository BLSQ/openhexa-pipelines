import pytest

from s3fs import S3FileSystem
from gcsfs import GCSFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem

import storage


S3_ENDPOINT_URL = "http://127.0.0.1:3000"


def test_filesystem():
    assert isinstance(storage.filesystem("/tmp/file.txt"), LocalFileSystem)
    assert isinstance(storage.filesystem("http://example.com/"), HTTPFileSystem)
    assert isinstance(storage.filesystem("s3://bucket/dir"), S3FileSystem)
    assert isinstance(storage.filesystem("gcs://bucket/dir"), GCSFileSystem)
    with pytest.raises(ValueError):
        storage.filesystem("bad://bucket/dir")
