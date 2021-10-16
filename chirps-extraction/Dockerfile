FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r /app/requirements.txt

COPY chirps.py .
COPY docker-entrypoint.sh .
COPY tests ./tests
COPY common.py .

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["chirps"]
