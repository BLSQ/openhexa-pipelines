# NDVI Extraction

```
Usage: python -m ndvi aggregate [OPTIONS]

Options:
  --start TEXT               start date of period to process  [required]
  --end TEXT                 end date of period to process  [required]
  --boundaries TEXT          aggregation boundaries  [required]
  --column-name TEXT         column name for boundary name  [required]
  --column-id TEXT           column name for boundary unique ID  [required]
  --csv TEXT                 output CSV file
  --db-user TEXT             database username
  --db-password TEXT         database password
  --db-host TEXT             database hostname
  --db-port INTEGER          database port
  --db-name TEXT             database name
  --db_table TEXT            database table
  --earthdata-username TEXT  nasa earthdata username  [required]
  --earthdata-password TEXT  nasa earthdata password  [required]
  --cache-dir TEXT           cache data directory
  --overwrite                overwrite existing data
  --help                     Show this message and exit.
```

## Usage

## Examples

```
ndvi aggregate \
    --start 2020-01-01 \
    --end 2021-01-01 \
    --boundaries localtests/input/bfa_districs.gpkg \
    --column-name "NOMDEP" \
    --column-id "DS_PCODE" \
    --csv localtests/output/ndvi.csv \
    --cache-dir localtests/cache
```

## Testing

Running pytest:

``` sh
docker-compose run pipeline test
```
