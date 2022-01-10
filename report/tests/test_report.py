import datetime
import json

from click.testing import CliRunner

import report


def test_report():
    runner = CliRunner()
    sample_date_1 = (
        datetime.datetime.now(tz=datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )
    sample_date_2 = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
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
            sample_date_1,
            "-l",
            sample_date_2,
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
        expected_content = f"""# Pipeline report
## Amazing pipeline
The content of this directory was created by a Pipeline run from OpenHexa.

Key facts:
- This run comes from a **Airflow** pipeline
- The run has `xb22` as identifier (use this for troubleshooting purposes)
- The pipeline was run on **{sample_date_1}**.
- The logical execution date of the pipeline is **{sample_date_2}**.

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
