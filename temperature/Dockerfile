FROM python:3.9.7

LABEL maintainer="jstilmant@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgdal-dev

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY temperature.py docker-entrypoint.sh common.py* /app/

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["temperature"]
