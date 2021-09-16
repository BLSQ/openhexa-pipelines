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
