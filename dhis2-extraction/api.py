import logging
import typing
from io import StringIO
from time import sleep
from typing import List, Union

import openhexa
import pandas as pd
from dhis2 import Api as BaseApi
from dhis2.exceptions import RequestException
from requests import Response

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
oh = openhexa.OpenHexaContext()
dag = oh.get_current_dagrun()


class MergedResponse:
    """
    Used to mirror the requests.Response interface for chunked requests

    Parameters
        ----------
        sample_response : requests.Response
            A response to use as basis (we will copy its headers and status code)
        content : bytes
            standard DHIS2.py API params
    """

    def __init__(self, sample_response: Response, content: bytes):
        self.headers = sample_response.headers
        self.status_code = sample_response.status_code
        self.content = content


class Api(BaseApi):
    def get(
        self,
        endpoint: str,
        file_type: str = "json",
        params: Union[dict, List[tuple]] = None,
        stream: bool = False,
        timeout: int = None,
        max_retries: int = 3,
    ) -> Response:

        retries = 0
        while retries < max_retries:

            try:

                r = self._make_request(
                    "get",
                    endpoint,
                    params=params,
                    file_type=file_type,
                    stream=stream,
                    timeout=timeout,
                )
                logger.info(f"Request URL: {r.url}")
                break

            # 1st try failed
            except RequestException as e:
                logger.warn("A request failed and is being retried")
                sleep(3)
                retries += 1
                if retries >= max_retries:
                    dag.log_message(
                        "ERROR", f"Connection to DHIS2 failed: error {e.code}"
                    )
                    raise

        return r

    def chunked_get(
        self,
        endpoint: str,
        *,
        params: dict,
        chunk_on: typing.Tuple[str, typing.Sequence[typing.Any]],
        chunk_size: int,
        **kwargs,
    ) -> typing.Union[Response, MergedResponse]:
        """
        Split a request in multiple chunks and merge the results.

        Parameters
        ----------
        endpoint : str
            The DHIS2 API endpoint
        params : dict
            standard DHIS2.py API params
        chunk_on : tuple
            a tuple of (parameter_name, parameter_values): the parameter that will determine the split
        chunk_size : int
            how many of "parameter_values" to handle by request

        Return
        ------
        str
            CSV content for now
        """

        if kwargs["file_type"] != "csv":
            raise ValueError("Only CSV file_type supported for now")

        chunk_parameter_name, chunk_parameter_values = chunk_on

        if len(chunk_parameter_values) == 0:
            raise ValueError("Cannot chunk requests with an empty list of values")
        elif len(chunk_parameter_values) < chunk_size:
            params[chunk_parameter_name] = chunk_parameter_values[0]
            return self.get(endpoint, params=params, **kwargs)

        df = pd.DataFrame()
        r = None
        for i in range(0, len(chunk_parameter_values), chunk_size):

            params[chunk_parameter_name] = chunk_parameter_values[i : i + chunk_size]

            # when large datasets are requested, DHIS2 may timeout
            # in that case, chunk on the longest dimension
            # 1st try: longest dimension is splitted in 2 parts
            # 2nd try: longest dimension is splitted in 4 parts
            # 3rd try: longest dimension is splitted in 8 parts
            MAX_RETRIES = 3
            TIMEOUT = 60
            retries = 0

            while retries < MAX_RETRIES:

                try:
                    r = self.get(endpoint, params=params, **kwargs)
                    break

                except RequestException as e:

                    if e.code == 504 and "analytics" in endpoint:

                        logger.warn(
                            "A request timed out and is beging chunked before retry"
                        )
                        logger.info("Request timed out")
                        retries += 1

                        parameter_values = split_params(
                            params=params.copy(),
                            chunk_parameter_name=chunk_parameter_name,
                            n_splits=2 ** (retries + 1),
                        )

                        r = self.chunked_get(
                            endpoint=endpoint,
                            params={chunk_parameter_name: None},
                            chunk_on=(chunk_parameter_name, parameter_values),
                            chunk_size=1,
                            file_type="csv",
                            timeout=TIMEOUT,
                        )
                        break

                    else:
                        dag.log_message(
                            "ERROR",
                            f"Connection to DHIS2 instance failed (HTTP Error {e.code}",
                        )
                        raise e

            df = pd.concat((df, pd.read_csv(StringIO(r.content.decode()))))

            # progress between 10 and 75% - downloading data is supposed to be
            # the longest step
            progress = 10 + ((i + 1) / len(chunk_parameter_values) * 65)
            dag.progress_update(round(progress))

        return MergedResponse(r, df.to_csv().encode("utf8"))


def split_list(src_list: list, n_splits: int = 2) -> List[list]:
    """Split a list into multiple parts."""
    length = len(src_list)
    return [
        src_list[i * length // n_splits : (i + 1) * length // n_splits]
        for i in range(n_splits)
    ]


def split_params(
    params: dict, chunk_parameter_name: str, n_splits: int = 2
) -> List[dict]:
    """Split request parameters into multiple parts."""
    chunks = []

    for param in params[chunk_parameter_name]:

        # get index of longest dimension
        max_length = 0
        longest = None
        for i, dimension in enumerate(param):
            key, elements = dimension.split(":")
            values = elements.split(";")
            if len(values) > max_length:
                longest = i

        dim = param[longest]
        key, elements = dim.split(":")
        values = elements.split(";")

        # split list of values in n_splits parts
        values_splitted = split_list(values, n_splits)

        for values in values_splitted:
            elements = ";".join([str(v) for v in values])
            # do not allow chunks with empty dimension
            if elements:
                dimension = f"{key}:{elements}"
                chunk = param.copy()
                chunk[longest] = dimension
                chunks.append(chunk)

    return chunks
