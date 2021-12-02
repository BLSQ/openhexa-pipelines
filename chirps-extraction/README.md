# Chirps extraction

The `chirps-extraction` pipeline downloads precipitation data from the [Climate Hazards Center (CHC)](https://www.chc.ucsb.edu/) through its [HTTP data repository](https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/) in order to aggregate zonal statistics based on contours provided by the user.

The program supports S3, GCS, and local paths for both inputs and outputs.

## Usage

This pipeline is meant to be run within a Docker container. The examples below use `docker-compose.yaml`, which is the
recommended approach for local development.

```
Usage: docker-compose run pipeline chirps [OPTIONS] COMMAND [ARGS]...

  Download and process CHIRPS data.

Options:
  --help  Show this message and exit.

Commands:
  download  Download raw precipitation data.
  extract   Compute zonal statistics.
```


## Data acquisition

```
Usage: chirps download [OPTIONS]

  Download raw precipitation data.

Options:
  --output-dir TEXT  output directory  [required]
  --start TEXT       start date  [required]
  --end TEXT         end date  [required]
  --overwrite        overwrite existing files
  --help             Show this message and exit.

```

The `download` subcommand downloads daily precipitation rasters for the african continent into a S3 directory.

### Example

``` sh
chirps download \
    --start 2017-02-01 \
    --end 2018-03-15 \
    --output-dir "s3://bucket/africa/daily"
```

## Zonal statistics

```
Usage: chirps extract [OPTIONS]

  Compute zonal statistics.

Options:
  --start TEXT        start date  [required]
  --end TEXT          end date  [required]
  --contours TEXT     path to contours  [required]
  --input-dir TEXT    chirps data directory  [required]
  --weekly TEXT       path to weekly output
  --monthly TEXT      path to monthly output
  --db-host TEXT      database hostname
  --db-port INTEGER   database port
  --db-name TEXT      database name
  --db-user TEXT      database username
  --db-password TEXT  database password
  --help              Show this message and exit.
```

The `extract` subcommand compute zonal statistics based on a contours file provided by the user. The output is a `csv` file written to a S3 path.

### Example

``` sh
docker-compose run pipeline chirps extract \
    --start 2017-01-01 \
    --end 2018-01-01 \
    --contours "s3://bucket/contours/bfa.geojson" \
    --input-dir "s3://bucket/africa/daily" \
    --weekly "s3://bucket/precipitation/bfa/weekly.csv" \
    --monthly "s3://bucket/precipitation/bfa/monthly.csv"
```

## Testing

Running pytest:

``` sh
docker-compose run pipeline pytest
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

## Code style

Our python code is linted using [`black`](https://github.com/psf/black), [`isort`](https://github.com/PyCQA/isort)
and [`autoflake`](https://github.com/myint/autoflake). We currently target the Python 3.9 syntax.

We use a [pre-commit](https://pre-commit.com/) hook to lint the code before committing. Make sure that `pre-commit` is
installed, and run `pre-commit install` the first time you check out the code. Linting will again be checked
when submitting a pull request.

You can run the lint tools manually using `pre-commit run --all`.
