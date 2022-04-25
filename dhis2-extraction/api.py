import logging
import typing
from io import StringIO

import pandas as pd
from dhis2 import Api as BaseApi
from requests import Response

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


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
            df = df.append(pd.read_csv(StringIO(r.content.decode())))

        return MergedResponse(r, df.to_csv().encode("utf8"))
