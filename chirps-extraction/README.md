# Description

```
Usage: chirps.py [OPTIONS] COMMAND [ARGS]...

  Download and process CHIRPS data.

Options:
  --help  Show this message and exit.

Commands:
  download  Download raw precipitation data.
  extract   Compute zonal statistics.
  test      Run test suite.
```

The `chirps-extraction` pipeline downloads precipitation data from the [Climate Hazards Center (CHC)](https://www.chc.ucsb.edu/) through its [HTTP data repository](https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/) in order to aggregate zonal statistics based on contours provided by the user.

## Data acquisition

```
Usage: chirps.py download [OPTIONS]

  Download raw precipitation data.

Options:
  --start INTEGER    start year
  --end INTEGER      end year
  --output-dir TEXT  output directory
  --overwrite        overwrite existing files
  --help             Show this message and exit.
```

The `download` subcommand downloads daily precipitation rasters for the african continent into a S3 directory.

### Example

``` sh
chirps.py download \
    --start 2017 \
    --end 2018 \
    --output-dir "s3://bucket/africa/daily"
```

## Zonal statistics

```
Usage: chirps.py extract [OPTIONS]

  Compute zonal statistics.

Options:
  --start INTEGER     start year
  --end INTEGER       end year
  --contours TEXT     path to contours file
  --input-dir TEXT    raw CHIRPS data directory
  --output-file TEXT  output directory
  --help              Show this message and exit.
```

The `extract` subcommand compute zonal statistics based on a contours file provided by the user. The output is a `csv` file written to a S3 path.

### Example

``` sh
chirps.py extract --start 2017 --end 2018 \
    --contours "s3://bucket/contours/bfa.geojson" \
    --input-dir "s3://bucket/africa/daily" \
    --output-file "s3://bucket/precipitation/bfa/chirps.csv" \
```

## Docker

The script can be run through Docker:

``` sh
docker run chirps:latest chirps --help
```

Running pytest:

``` sh
docker run chirps:latest pytest tests/
```

## S3 credentials

S3 access can be configured by passing the following environment variables.

Read by `s3fs`:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_SESSION_TOKEN`
* `AWS_DEFAULT_REGION`
* See [boto3 doc](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables)

Read by `rasterio` (via `GDAL`):

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_SESSION_TOKEN`
* `AWS_DEFAULT_REGION`
* `AWS_S3_ENDPOINT`
* `AWS_HTTPS=YES`
* `AWS_VIRTUAL_HOSTING=TRUE`
* `AWS_NO_SIGN_REQUEST=YES`
* See [GDAL doc](https://gdal.org/user/virtual_file_systems.html#vsis3-aws-s3-files)

## Local

The program only support S3 paths for both inputs and outputs. In order to run the script locally, a local S3 server must be setup first. For instance with [MinIO](https://min.io/). Environment variables must be passed to the Docker image so that both `boto3` and `GDAL` are able to connect to the local server.

``` sh
# create minio directory and a "chirps" bucket
mkdir data/chirps
minio_server --address ":9000"

# add contours file to the "chirps" bucket
mkdir -p data/chirps/input/contours/bfa.geojson
cp tests/bfa.geojson data/chirps/input/contours/

# run "chirps download"
podman run \
    --network="host" \
    -e AWS_ACCESS_KEY_ID="minioadmin" \
    -e AWS_SECRET_ACCESS_KEY="minioadmin" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -e AWS_S3_ENDPOINT="127.0.0.1:9000" \
    -e AWS_HTTPS="NO" \
    -e AWS_VIRTUAL_HOSTING="FALSE" \
    chirps:latest chirps download \
        --start 2017 \
        --end 2018 \
        --output-dir "s3://chirps/input/africa" \

# run "chirps extract"
podman run \
    --network="host" \
    -e AWS_ACCESS_KEY_ID="minioadmin" \
    -e AWS_SECRET_ACCESS_KEY="minioadmin" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -e AWS_S3_ENDPOINT="127.0.0.1:9000" \
    -e AWS_HTTPS="NO" \
    -e AWS_VIRTUAL_HOSTING="FALSE" \
    -e GDAL_DISABLE_READDIR_ON_OPEN="YES" \
    chirps:latest chirps extract \
        --start 2017 \
        --end 2018 \
        --contours "s3://chirps/input/contours/bfa.geojson" \
        --input-dir "s3://chirps/input/africa" \
        --output-file "s3://chirps/output/bfa.csv"
```
