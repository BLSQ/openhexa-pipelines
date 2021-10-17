FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY dhis2extract.py period.py docker-entrypoint.sh common.py? .

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dhis2extract"]
