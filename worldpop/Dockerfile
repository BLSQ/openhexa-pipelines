FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgdal-dev

COPY requirements.txt .
RUN pip install -r /app/requirements.txt
RUN pip install git+https://github.com/BLSQ/openhexa-sdk.git@cee07535edf5bb7b4f39e14cf83182f5e62e1c82

COPY worldpop.py docker-entrypoint.sh common.py* /app/

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["worldpop"]
