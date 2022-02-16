FROM python:3.9.7

LABEL maintainer="jstilmant@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY temperature.py docker-entrypoint.sh common.py? .

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["temperature"]
