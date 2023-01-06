#!/usr/bin/env python
import argparse
import base64
import datetime
import json
import logging
import sys

import papermill as pm

# comon is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError as e:
    print(f"Unexpected {e=}, {type(e)=}")
    # ignore import error -> work anyway (but define logging)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("papermill_app")

# import fuse mount script _after_ env variables injection
# these files come from blsq notebooks image
sys.path.insert(1, "/home/jovyan/.fuse")
import fuse_mount  # noqa: F401, E402

parser = argparse.ArgumentParser(description="Papermill pipeline")
parser.add_argument(
    "-i",
    dest="in_nb",
    action="store",
    type=str,
    help="Input notebook to use",
    required=True,
)
parser.add_argument(
    "-o",
    dest="out_nb",
    action="store",
    type=str,
    help="Output notebook to generate",
)
parser.add_argument(
    "-p",
    dest="parameters",
    action="store",
    type=str,
    help="Pipeline parameters",
    default="",
)

args = parser.parse_args()

logger.info("source parameters: %s", args.parameters)
if args.parameters:
    parameters = json.loads(base64.b64decode(args.parameters).decode())
else:
    parameters = {}


def dumb_cast(v):
    # should we parse datetime?
    if v in ("False", "false", "FALSE"):
        return False
    if v in ("True", "true", "TRUE"):
        return True
    if v in ("None", "null", "NULL"):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v


parameters = {k: dumb_cast(v) for k, v in parameters.items()}

# inject execution date, if present
execution_date = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")
if args.out_nb:
    out_notebook = args.out_nb.replace("%DATE", execution_date)
else:
    out_notebook = None

logger.info(
    "Papermill pipeline start, in %s, out %s, parameters %s",
    args.in_nb,
    out_notebook,
    parameters,
)

pm.execute_notebook(args.in_nb, out_notebook, parameters=parameters, progress_bar=False)

# umount at the end
import fuse_umount  # noqa: F401, E402
