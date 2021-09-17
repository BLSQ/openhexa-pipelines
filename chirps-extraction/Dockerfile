FROM python:3.9.7

LABEL maintainer="yforget@bluesquarehub.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/app/venv

RUN mkdir /app
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY chirps.py /app/
COPY tests /app/tests

WORKDIR /app

ENTRYPOINT ["python", "-m"]
