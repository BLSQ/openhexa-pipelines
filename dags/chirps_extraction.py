from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.bash import BashOperator


test_var = "A" # Variable.get("SECRET_TEST_VAR")


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='0 1 27 * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = KubernetesPodOperator(
        namespace='hexa-airflow',
        image="blsq/chirps-extraction",
        arguments="chirps download --start 2020 --end 2020 --output-dir s3://hexa-demo-pipeline-chirps/download/".split(" "),
        name="chirps-download-pod",
        task_id="chirps-download",
        env_vars={"TEST_VAR": test_var},
        in_cluster=True,
        get_logs=True,
    )
    extract = KubernetesPodOperator(
        namespace='hexa-airflow',
        image="blsq/chirps-extraction",
        arguments="chirps extract --start 2020 --end 2020 --input-dir s://hexa-demo-pipeline-chirps/download/ --output-file s3://hexa-demo-pipeline-chirps/extract.csv  --contours s3://hexa-demo-pipeline-chirps/world/BFA.geo.json ".split(" "),
        name="chirps-extract-pod",
        task_id="chirps-extract",
        env_vars={"TEST_VAR": test_var},
        in_cluster=True,
        get_logs=True,
    )

    download >> extract
