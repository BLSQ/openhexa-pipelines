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


def filesystem(path):
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
    if "://" in path:
        protocol = path.split("://")[0]
        if protocol == "s3":
            return S3FileSystem(
                client_kwargs={"endpoint_url": os.environ.get("AWS_S3_ENDPOINT")}
            )
        elif protocol == "gcs":
            return GCSFileSystem()
        elif protocol == "http" or protocol == "https":
            return HTTPFileSystem()
        else:
            raise ValueError(f"Protocol {protocol} not supported.")
    else:
        return LocalFileSystem()


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
