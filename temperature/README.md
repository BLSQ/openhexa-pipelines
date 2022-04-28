# Temperature

The `temperature` pipeline downloads data from NOAA CPC's [Global Daily
Temperature](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html) and
aggregates the value in time (weekly, monthly) and in space (administrative
boundary). Output data is exported to individual CSV files and/or Postgres
tables.

## Usage

```
Usage: python -m temperature [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  aggregate  Compute weekly zonal statistics.
  daily      Compute daily zonal statistics.
  sync       Sync yearly temperature data sets.
```

### Download/sync raw data

The subcommand `sync` checks for available products in CPC servers and download the
missing files into a data directory.

```
Usage: python -m temperature sync [OPTIONS]

  Sync yearly temperature data sets.

Options:
  --start TEXT       start date of period to sync  [required]
  --end TEXT         end date of period to sync  [required]
  --output-dir TEXT  output data directory  [required]
  --overwrite        overwrite existing files
  --help             Show this message and exit.
```

The `--overwrite` flag is required to replace outdated products with their
up-to-date versions.

### Compute daily zonal statistics

The subcommand `daily` computes daily zonal statistics (mean, min, max) for each
boundary geometry and save the result into a netCDF (`*.nc`) file.

```
Usage: python -m temperature daily [OPTIONS]

  Compute daily zonal statistics.

Options:
  --start TEXT            start date of analysis  [required]
  --end TEXT              end date of analysis  [required]
  --boundaries TEXT       aggregation boundaries  [required]
  --boundaries-name TEXT  column with boundary name  [required]
  -i, --input-dir TEXT    input data directory  [required]
  -o, --output-file TEXT  output dataset  [required]
  --overwrite             overwrite files
  --help                  Show this message and exit.
```

Boundaries can be provided as GeoJSON or Geopackage files. They will be
automatically reproject to WGS84 if needed. The column containing a unique
boundary name or identifier must be provided by the user.

### Compute weekly and monthly extracts

The subcommand `aggregate` computes weekly and monthly extracts from the daily
zonal statistics computed with the `daily` subcommand. Output can be a CSV or a
Postgres table.

```
Usage: python -m temperature aggregate [OPTIONS]

  Compute weekly zonal statistics.

Options:
  --daily-file TEXT             daily zonal statistics  [required]
  --frequency [weekly|monthly]  aggregation frequency  [required]
  --output-file-tmin TEXT       output csv file
  --output-file-tmax TEXT       output csv file
  --output-table-tmin TEXT      output SQL table
  --output-table-tmax TEXT      output SQL table
  --db-user TEXT                DB username
  --db-password TEXT            DB password
  --db-host TEXT                DB hostname
  --db-port INTEGER             DB port
  --db-name TEXT                DB name
  --help                        Show this message and exit.
```