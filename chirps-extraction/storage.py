import os
import logging

import fsspec
from fsspec import registry, AbstractFileSystem, get_filesystem_class
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


def open(path, *args):
    """Open a file."""
    fs = filesystem(path)
    logging.debug(f"Opening {path}")
    return fs.open(_no_protocol(path), *args)


def exists(path):
    """Check path existence."""
    fs = filesystem(path)
    logging.debug(f"Checking existence of {path}")
    return fs.exists(_no_protocol(path))


def glob(path):
    """Glob a path pattern."""
    fs = filesystem(path)
    logging.debug(f"Globbing {path}")
    results = fs.glob(_no_protocol(path))
    # re-add protocol prefix
    if "://" in path:
        protocol = path.split("://")[0]
        return [f"{protocol}://{res}" for res in results]
    return results


def makedirs(path):
    """Create directories recursively."""
    fs = filesystem(path)
    logging.debug(f"Creating directories at {path}")
    return fs.makedirs(_no_protocol(path), exist_ok=True)


def size(path):
    """Get size of a file in bytes."""
    fs = filesystem(path)
    logging.debug(f"Getting file size of {path}")
    return fs.du(_no_protocol(path))


def put(local_path, remote_path, recursive=False):
    """Upload a local file to a remote location.

    Upload a tree of files if recursive=True.
    """
    fs = filesystem(remote_path)
    logging.debug(f"Uploading {local_path} to {remote_path}")
    return fs.put(local_path, _no_protocol(remote_path), recursive=recursive)


def get(remote_path, local_path, recursive=False):
    """Download a remote file into a local location.

    Download a tree of files if recursive=True.
    """
    fs = filesystem(remote_path)
    logging.debug(f"Downloading {remote_path} into {local_path}")
    return fs.get(_no_protocol(remote_path), local_path, recursive=recursive)
