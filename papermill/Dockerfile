FROM jupyter/datascience-notebook:92ce0af9989f

USER root

# System libraries
RUN apt-get update

# TBD: seem useful in the long term, but how to pass the perms?
# s3fs fuse
# RUN sudo apt install -y s3fs

# Add locales
RUN locale-gen fr_FR fr_FR.UTF-8 es_ES es_ES.UTF-8 de_DE de_DE.UTF-8
RUN update-locale

# Install AWS CLI tools
RUN mkdir -p /usr/local/src/awcli \
    && cd /usr/local/src/awcli \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && cd $HOME \
    && rm -rf /usr/local/src/awcli

# Use mamba instead of conda
RUN conda install --quiet --yes mamba -c conda-forge

# Python packages, in multiple steps for Docker layering (mamba preferred method)
# TODO: unpin s3contents, fsspec, gcsfs & s3fs once https://github.com/danielfrg/s3contents/issues/122 is fixed
RUN mamba install --yes \
    'dask-ml=1.*' \
    'descartes=1.*' \
    'fsspec=0.9.0' \
    'gcsfs=0.8.0' \
    'geoalchemy2=0.*' \
    'geopandas=0.*' \
    'hybridcontents=0.*' \
    'ipywidgets=7.*' \
    'lxml=4.*' \
    'mapclassify=2.*' \
    'nbresuse=0.*' \
    'plotly=4.*' \
    'psycopg2=2.*' \
    'rapidfuzz=0.*' \
    'rasterstats=0.*' \
    's3contents=0.6.0' \
    's3fs=0.4.2' \
    'papermill==2.*' \
    && mamba clean --yes --all

# Python packages (pip - for what is not available in mamba)
RUN pip install \
    'cowsay==3.*' \
    'fake-useragent==0.*' \
    'hdx-python-api==4.*' \
    'tabpy==2.*' \
    'sentry-sdk'

RUN mkdir /app/
COPY __init__.py app.py common.py? /app/
WORKDIR /app/
ENTRYPOINT ["/app/app.py"]
CMD ["--help"]
