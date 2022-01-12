# DHIS2 Extraction

The `dhis2-extraction` pipeline downloads data from a DHIS2 instance via the API and transforms it into formatted CSV files.

The program supports S3, GCS, and local paths for outputs.

```
Usage: dhis2extract.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download   Download data from a DHIS2 instance via its web API.
  transform  Transform raw data from DHIS2 into formatted CSV files.
```

## Download metadata and raw data

`dhis2extract download` downloads raw data from a DHIS2 instance for a given set of input parameters.

### Usage

At least one temporal dimension is required: either `period` or `start-date` and `end-date`.

At least one data dimension is required: either `data-element`, `data-element-group`, `indicator`, or `indicator-group`. `dataset` can be used to fetch all the data elements belonging to a given DHIS2 dataset.

At least one org unit dimension is required: either `org-unit`, `org-unit-group` or `org-unit-level`.

Dimension parameters (periods, org units, datasets, data elements, indicators, programs, etc) can be repeated any number of times.

With the `--metadata-only` flag, only the metadata tables are downloaded.

The program may perform requests to 3 different DHIS2 API endpoints: `api/dataValueSet`, `api/analytics`, or `api/analytics/rawData`:

**Exporting raw data values from the analytics tables** (default)  
`--mode analytics-raw`  
The `api/analytics/rawData` endpoint from DHIS2 is used.
Data values are exported from the analytics tables as raw data values, i.e. no temporal aggregation is performed.
Note: data for children of the provided org. units will also be collected.

**Exporting aggregated data values from the analytics tables**  
`--mode analytics`  
The `api/analytics` endpoint from DHIS2 is used.
Data values are exported from the analytics tables and aggregated according to the `period` dimension, *e.g.* data will be aggregated per year if periods were provided as DHIS2 years.

**Exporting raw data values**  
`--mode raw`  
The `api/dataValueSet` endpoint from DHIS2 is used.
Raw data values are exported (no analytics tables). No aggregation is performed. This is useful if you want to export data that is not yet included in the analytics tables.
Note: data values are supposed to be extracted per dataset. If data elements UIDs are provided, the entire dataset to which the data element belong will be downloaded.

```
Usage: dhis2extract.py download [OPTIONS]

  Download data from a DHIS2 instance via its web API.

Options:
  -i, --instance TEXT             DHIS2 instance URL.  [required]
  -u, --username TEXT             DHIS2 username.  [required]
  -p, --password TEXT             DHIS2 password.  [required]
  -o, --output-dir TEXT           Output directory.  [required]
  -s, --start TEXT                Start date in ISO format.
  -e, --end TEXT                  End date in ISO format.
  -pe, --period TEXT              DHIS2 period. *
  -ou, --org-unit TEXT            Organisation unit UID. *
  -oug, --org-unit-group TEXT     Organisation unit group UID. *
  -lvl, --org-unit-level INTEGER  Organisation unit level. *
  -ds, --dataset TEXT             Dataset UID. *
  -de, --data-element TEXT        Data element UID. *
  -deg, --data-element-group TEXT
                                  Data element group UID. *
  -in, --indicator TEXT           Indicator UID. *
  -ing, --indicator-group TEXT    Indicator group UID. *
  -aoc, --attribute-option-combo TEXT
                                  Attribute option combo UID. *
  -coc, --category-option-combo TEXT
                                  Category option combo UID. *
  -prg, --program TEXT            Program UID. *
  --from-json TEXT                Load parameters from a JSON file.
  --children / --no-children      Include children of selected org units.
  -m, --mode TEXT                 Request mode.
  --metadata-only                 Only download metadata.
  --overwrite                     Overwrite existing file.
  --help                          Show this message and exit.

  (*) Can be provided multiple times.
```

### Examples

```
dhis2extract download \
    --instance play.dhis2.org/2.34.7 \
    --username <dhis2_username> \
    --password <dhis2_password> \
    --output-dir "s3://<bucket>/dhis2/extract" \
    --start 2020-01-01 --end 2021-31-12 \
    --org-unit-level 4 \
    -de "kfN3vElj7in" -de "RTSp1yfraiu" -de "v2wejz8CGLb"
```

## Transform data into formatted CSV files

`dhis2extract transform` processes the raw data downloaded with `dhis2extract transform` into formatted CSV files. 

The following files are created in `output-dir`:

* Data table
    * `extract.csv`
* Metadata tables
    * `metadata/metadata/organisation_units.csv`
    * `metadata/organisation_units.gpkg`
    * `metadata/organisation_unit_groups.csv`
    * `metadata/data_elements.csv`
    * `metadata/indicators.csv`
    * `metadata/indicator_groups.csv`
    * `metadata/datasets.csv`
    * `metadata/programs.csv`
    * `metadata/category_option_combos.csv`
    * `metadata/category_combos.csv`
    * `metadata/category_options.csv`
    * `metadata/categories.csv`
  
`extract.csv` is a table with one row per data element, org. unit and period. The following columns are created:

* `dx_uid`, `dx_name` : data element or indicator UID and name
* `dx_type` : data element or indicator
* `coc_uid`, `coc_name` : category option combo UID and name
* `period` : period in DHIS2 format
* `ou_uid` : org. unit UID
* `level_{1,2,3,4,5}_uid`, `level_{1,2,3,4,5}_name` : org. unit UID and name for each hierarchical level

### Usage

```
Usage: dhis2extract.py transform [OPTIONS]

  Transform raw data from DHIS2 into formatted CSV files.

Options:
  -i, --input-dir TEXT   Input directory.
  -o, --output-dir TEXT  Output directory.
  --overwrite            Overwrite existing files.
  --help                 Show this message and exit.
```

### Examples

```
dhis2extract transform \
    -i "s3://<bucket>/dhis2/extract/raw" \
    -o "s3://<bucket>/dhis2/extract"
```

## Testing

Running pytest:

``` sh
docker-compose run pipeline test
```
