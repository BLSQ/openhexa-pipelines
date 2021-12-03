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

The program uses the appropriate endpoint according to the `--aggregate` and `--analytics` flags:

* If `--aggregate`, the Analytics API is used to aggregate data element values per hierarchical level and period ;
* If `--no-analytics`, the Analytics API is avoided -- this can be useful to download data that have not been incoprated to the analytics data tables yet.

At least one temporal dimension is required: either `period` or `start-date` and `end-date`.

At least one data dimension is required: either `data-element`, `data-element-group`, `indicator`, or `indicator-group`. `dataset` can be used to fetch all the data elements belonging to a given DHIS2 dataset.

At least one org unit dimension is required: either `org-unit`, `org-unit-group` or `org-unit-level`.

Dimension parameters (periods, org units, datasets, data elements, indicators, programs, etc) can be repeated any number of times.

With the `--skip` flag, only the metadata tables are downloaded.

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
  -prg, --program TEXT            Program UID. *
  --from-json TEXT                Load parameters from a JSON file.
  --children / --no-children      Include children of selected org units.
  --aggregate / --no-aggregate    Aggregate using Analytics API.
  --analytics / --no-analytics    Use the Analytics API.
  --skip                          Only download metadata.
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

`dhis2extract transform` processes the raw data downloaded with `dhis2extract transform` into formatted CSV files. The following files are created in `output-dir`:

* Metadata tables
    * `organisation_units.csv`
    * `organisation_units.gpkg`
    * `organisation_unit_groups.csv`
    * `data_elements.csv`
    * `datasets.csv`
    * `category_option_combos.csv`
    * `category_combos.csv`
    * `category_options.csv`
    * `categories.csv`
* Data table
    * `extract.csv`

`extract.csv` is a table with the following columns: `DX_UID`, `COC_UID`, `AOC_UID`, `PERIOD`, `OU_UID` and `VALUE`.

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
