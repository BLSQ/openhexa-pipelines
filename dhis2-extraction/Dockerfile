FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt
RUN pip install git+https://github.com/BLSQ/openhexa-sdk.git@cee07535edf5bb7b4f39e14cf83182f5e62e1c82

COPY dhis2extract.py period.py api.py docker-entrypoint.sh common.py* .

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dhis2extract"]
