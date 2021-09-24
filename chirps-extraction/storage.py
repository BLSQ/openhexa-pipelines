import os
import logging

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


def _no_protocol(path):
    """Get path without protocol prefix.

    Most s3fs and gcsfs function expect it as input.

    Parameters
    ----------
    path : str
        Full path with or without protocol.

    Return
    ------
    str
        Path without protocol prefix.
    """
    if "://" in path:
        return path.split("://")[1]
    return path


def filesystem(target_path):
    """Guess filesystem from path.

    As of now 4 filesystems are supported: S3, GCS, HTTP, and local.

    Parameters
    ----------
    path : str
        Path to file or directory.

    Return
    ------
    FileSystem
        Appropriate FileSystem object.
    """

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

    class LoggedFileSystem(fs_class):
        def glob(self, path, *, re_add_protocol=False, **kwargs):
            results = super().glob(path, **kwargs)
            if re_add_protocol:
                if "://" in path:
                    protocol = path.split("://")[0]
                    results = [f"{protocol}://{res}" for res in results]
            return results

        def dirname(self, path):
            return os.path.dirname(self._strip_protocol(path))

    return LoggedFileSystem(client_kwargs=client_kwargs)
