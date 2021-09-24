import os
import subprocess
import time

import pytest
import requests
import responses


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture()
def moto_server():
    os.environ["AWS_S3_ENDPOINT"] = "http://localhost:3000"

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    if "AWS_S3_ENDPOINT" not in os.environ:
        os.environ["AWS_S3_ENDPOINT"] = "http://127.0.0.1:3000"

    p = subprocess.Popen(["moto_server", "s3", "-p", "3000"])

    timeout = 5
    while timeout > 0:
        try:
            r = requests.get(os.environ["AWS_S3_ENDPOINT"])
            if r.ok:
                break
        except:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    yield
    p.terminate()
    p.wait()
