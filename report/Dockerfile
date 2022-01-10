FROM python:3.9.7

LABEL maintainer="pvanliefland@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY docker-entrypoint.sh .
COPY report.py .

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["report"]
