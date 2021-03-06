FROM jupyter/datascience-notebook:4b830d5897d8

USER root

# System libraries
RUN apt-get update

# FUSE Amazon S3
RUN sudo apt install -y s3fs

# FUSE Google Cloud Storage
WORKDIR /tmp
RUN wget -q "https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v0.40.0/gcsfuse_0.40.0_amd64.deb" -O "gcsfuse.deb"
RUN apt install -y ./gcsfuse.deb
RUN rm ./gcsfuse.deb
WORKDIR $HOME

# Add locales
RUN locale-gen fr_FR fr_FR.UTF-8 es_ES es_ES.UTF-8 de_DE de_DE.UTF-8
RUN update-locale

# For rgdal & sf (R)
RUN apt-get install -y libgdal-dev libudunits2-dev

# Install AWS CLI tools
RUN mkdir -p /usr/local/src/awcli
WORKDIR /usr/local/src/awcli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
WORKDIR $HOME
RUN rm -rf /usr/local/src/awcli

# Use mamba instead of conda
RUN conda install --quiet --yes mamba -c conda-forge

# R packages (mamba + conda-forge)
RUN mamba install --yes -c conda-forge \
    'r-arrow=7.*' \
    'r-aws.s3=0.*' \
    'r-geojsonio=0.*' \
    'r-getpass=0.*' \
    'r-ggmap=3.*' \
    'r-ggthemes=4.*' \
    'r-hmisc=4.*' \
    'r-maptools=1.*' \
    'r-plotly=4.*' \
    'r-raster=3.*' \
    'r-readxl=1.*' \
    'r-rcolorbrewer=1.*' \
    'r-rgdal=1.*' \
    'r-rgeos=0.*' \
    'r-rgooglemaps=1.*' \
    'r-rjava=1.*' \
    'r-rjson=0.*' \
    'r-rpostgres=1.*' \
    'r-sf=1.*' \
    'r-styler=1.*' \
    'r-tidyverse=1.*' \
    'r-viridis=0.*' \
    && mamba clean --yes --all

# R packages (cran)
RUN R -e "install.packages('GISTools', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"
RUN R -e "install.packages('OpenStreetMap', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"
RUN R -e "install.packages('isotree', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"
RUN R -e "install.packages('MMWRweek', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"
RUN R -e "install.packages('imputeTS', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"
RUN R -e "install.packages('tsbox', dependencies=TRUE, quiet=TRUE, repos='https://cran.r-project.org/')"

# Python packages (mamba + conda-forge)
RUN mamba install --yes -c conda-forge \
    'black=22.*' \
    'dask-ml=2022.*' \
    'descartes=1.*' \
    'fsspec=2022.*' \
    'gcsfs=2022.*' \
    'geoalchemy2=0.*' \
    'geopandas=0.*' \
    'hybridcontents=0.*' \
    'ipywidgets=7.*' \
    'isort=5.*' \
    'lxml=4.*' \
    'mapclassify=2.*' \
    'nbresuse=0.*' \
    'netCDF4=1.*' \
    'papermill==2.*' \
    'plotly=5.*' \
    'psycopg2=2.*' \
    'pyarrow=7.*' \
    'rapidfuzz=2.*' \
    'rasterstats=0.*' \
    'rpy2=3.*' \
    's3contents=0.*' \
    's3fs=2022.*' \
    'slack_sdk=3.*' \
    'voila=0.*' \
    'xarray=2022.*' \
    'zarr=2.*' \
    && mamba clean --yes --all

# Python packages (pip)
RUN pip install \
    'cowsay==4.*' \
    'fake-useragent==0.*' \
    'hdx-python-api==5.*' \
    'tabpy==2.*' \
    'sentry_sdk'

# Jupyter extensions
RUN mamba install -c conda-forge --yes \
    'jupyter-dash=0.*' \
    'jupyter-resource-usage=0.*' \
    'jupyter-server-proxy=3.*' \
    'jupyterlab_code_formatter=1.*' \
    && mamba clean --yes --all

# Build Jupyterlab - needed for extensions
RUN jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf "/home/${NB_USER}/.cache/yarn"

# copy specific files to the image
COPY fuse/fuse.conf /etc/.
COPY __init__.py app.py common.py* fuse/fuse_mount.py fuse/fuse_umount.py /home/jovyan/
WORKDIR /home/jovyan/
ENTRYPOINT ["/home/jovyan/app.py"]
CMD ["--help"]
