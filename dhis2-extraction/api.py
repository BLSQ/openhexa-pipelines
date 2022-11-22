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
            params[chunk_parameter_name] = chunk_parameter_values

            return self.get(endpoint, params=params, **kwargs)

        df = pd.DataFrame()
        r = None
        for i in range(0, len(chunk_parameter_values), chunk_size):
            params[chunk_parameter_name] = chunk_parameter_values[i : i + chunk_size]
            r = self.get(endpoint, params=params, **kwargs)
            logger.info(f"Request URL: {r.url}")
            df = pd.concat((df, pd.read_csv(StringIO(r.content.decode()))))

            # progress between 10 and 75% - downloading data is supposed to be
            # the longest step
            progress = 10 + ((i + 1) / len(chunk_parameter_values) * 65)
            dag.progress_update(round(progress))

        return MergedResponse(r, df.to_csv().encode("utf8"))
