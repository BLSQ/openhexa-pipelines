import os
import logging
import typing

from fsspec import AbstractFileSystem
from s3fs import S3FileSystem
from gcsfs import GCSFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def add_logging(
    fs: AbstractFileSystem, methods: typing.Sequence[str]
) -> AbstractFileSystem:
    for method_name in methods:
        method = getattr(fs, method_name)

        def method_with_logging(*args, **kwargs):
            args_log = ", ".join(args)
            kwargs_log = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            logger.info(
                f"Calling {method} with args {args_log} and kwargs {kwargs_log}"
            )
            method(*args, **kwargs)

        setattr(fs, method_name, method_with_logging)

    return fs


def filesystem(target_path: str) -> AbstractFileSystem:
    client_kwargs = {}
    if "://" in target_path:
        target_protocol = target_path.split("://")[0]
        if target_protocol == "s3":
            fs_class = S3FileSystem
            client_kwargs = {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}
        elif target_protocol == "gcs":
            fs_class = GCSFileSystem
        elif target_protocol == "http" or target_protocol == "https":
            fs_class = HTTPFileSystem
        else:
            raise ValueError(f"Protocol {target_protocol} not supported.")
    else:
        fs_class = LocalFileSystem

    return fs_class(client_kwargs=client_kwargs)
    return add_logging(fs_class(client_kwargs=client_kwargs), ["glob", "put"])
