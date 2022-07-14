# ERA5-Land pipeline

```
Usage: python -m era5 aggregate [OPTIONS]

Options:
  --start TEXT         start date of period to process  [required]
  --end TEXT           end date of period to process  [required]
  --boundaries TEXT    aggregation boundaries  [required]
  --column-name TEXT   column name for boundary name  [required]
  --column-id TEXT     column name for boundary unique ID  [required]
  --cds-variable TEXT  CDS variable of interest  [required]
  --agg-function TEXT  spatial aggregation function
  --hours TEXT         hour of the day (or ALL)
  --csv TEXT           output CSV file
  --db-user TEXT       database username
  --db-password TEXT   database password
  --db-host TEXT       database hostname
  --db-port INTEGER    database port
  --db-name TEXT       database name
  --db_table TEXT      database table
  --cds-api-key TEXT   CDS api key  [required]
  --cds-api-uid TEXT   CDS user ID  [required]
  --cache-dir TEXT     cache data directory
  --overwrite          overwrite existing data
  --help               Show this message and exit.
```

## Usage

## Examples

Daily precipitation:

```
python -m era5 aggregate \
    --start 2017-01-01 \
    --end 2017-03-15 \
    --boundaries localtests/input/bfa_districs.gpkg \
    --column-name "NOMDEP" \
    --column-id "DS_PCODE" \
    --cds-variable "total_precipitation" \
    --hours "ALL" \
    --csv "localtests/output/total_precipitation.csv" \
    --cds-api-key "b83aa1b0-ec21-4190-986e-9da80aee4912" \
    --cds-api-uid "145677" \
    --cache-dir "localtests/cache/" \
    --overwrite
```

Daily temperature:

```
python -m era5 aggregate \
    --start 2017-01-01 \
    --end 2017-03-15 \
    --boundaries localtests/input/bfa_districs.gpkg \
    --column-name "NOMDEP" \
    --column-id "DS_PCODE" \
    --cds-variable "2m_temperature" \
    --hours "06:00,12:00,18:00" \
    --csv "localtests/output/2m_temperature.csv" \
    --cds-api-key "b83aa1b0-ec21-4190-986e-9da80aee4912" \
    --cds-api-uid "145677" \
    --cache-dir "localtests/cache/" \
    --overwrite
```

## Testing

Running pytest:

``` sh
docker-compose run pipeline test
```
