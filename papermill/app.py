#!/usr/bin/env python

import argparse
import logging

import papermill as pm

# comon is a script to set parameters on production
try:
    import common  # noqa: F401
except ImportError as e:
    # ignore import error -> work anyway (but define logging)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("papermill_app")

parser = argparse.ArgumentParser(description="Papermill pipeline")
parser.add_argument(
    "--in", dest="in_nb", action="store", type=str, help="Input notebook to use"
)
parser.add_argument(
    "--out", dest="out_nb", action="store", type=str, help="Output notebook to generate"
)
parser.add_argument(
    "-p", dest="parameters", action="append", type=str, help="Pipeline parameters"
)

args = parser.parse_args()
parameters = dict([e.split("=", 1) for e in args.parameters])


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
logger.info(
    "Papermill pipeline start, in %s, out %s, parameters %s",
    args.in_nb,
    args.out_nb,
    parameters,
)
pm.execute_notebook(args.in_nb, args.out_nb, parameters=parameters, progress_bar=False)
