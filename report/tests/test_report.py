import datetime
import json

from click.testing import CliRunner

import report


def test_report():
    runner = CliRunner()
    sample_date = datetime.datetime.now().replace(microsecond=0).isoformat()
    result = runner.invoke(
        report.cli,
        [
            "report",
            "-o",
            "/code/some_dir",
            "-t",
            "Airflow",
            "-r",
            "xb22",
            "-d",
            sample_date,
            "-l",
            sample_date,
            "-h",
            "Amazing pipeline",
            "-i",
            "This pipeline does wonders!",
            "-c",
            json.dumps({"foo": "bar"}),
        ],
    )
    assert result.exit_code == 0

    with open("/code/some_dir/README.md") as f:
        actual_content = f.read()
        expected_content = f"""
# Pipeline report
## Amazing pipeline
The content of this directory was created by a Pipeline run from OpenHexa.

Key facts:
- This run comes from a **Airflow** pipeline
- The run has `xb22` as identifier (use this for troubleshooting purposes)
- The pipeline was run on **{sample_date}**.
- The logical execution date of the pipeline is **{sample_date}**.

## Additional info
This pipeline does wonders!

## Used configuration
```
{{
  "foo": "bar"
}}
```
    """

    assert actual_content == expected_content
