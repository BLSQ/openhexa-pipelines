# DHIS2 Extraction

The `dhis2-extraction` pipeline downloads data from a DHIS2 instance via the API and transforms it into formatted CSV files.

The program supports S3, GCS, and local paths for outputs.

## Usage

```
Usage: dhis2extract.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download   Download data from a DHIS2 instance via its web API.
  transform  Transform raw data from DHIS2 into formatted CSV files.
```

### Download metadata and raw data

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
  -pe, --period TEXT              DHIS2 period.
  -ou, --org-unit TEXT            Organisation unit UID.
  -oug, --org-unit-group TEXT     Organisation unit group UID.
  -lvl, --org_unit_level INTEGER  Organisation unit level.
  -ds, --dataset TEXT             Dataset UID.
  -de, --data-element TEXT        Data element UID.
  -deg, --data-element-group TEXT
                                  Data element group UID.
  -in, --indicator TEXT           Indicator UID.
  -ing, --indicator-group TEXT    Indicator group UID.
  -aoc, --attribute-option-combo TEXT
                                  Attribute option combo UID.
  -prg, --program TEXT            Program UID.
  --from-json TEXT                Load parameters from a JSON file.
  --children / --no-children      Include childrens of selected org units.
  --aggregate / --no-aggregate    Aggregate using Analytics API.
  --analytics / --no-analytics    Use the Analytics API.
  --skip                          Only download metadata.
  --overwrite                     Overwrite existing file.
  --help                          Show this message and exit.
```

### Transform data into formatted CSV files

```
Usage: dhis2extract.py transform [OPTIONS]

  Transform raw data from DHIS2 into formatted CSV files.

Options:
  -o, --output-dir TEXT  Output directory.
  --overwrite            Overwrite existing files.
  --help                 Show this message and exit.
```
