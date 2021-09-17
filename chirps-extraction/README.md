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
