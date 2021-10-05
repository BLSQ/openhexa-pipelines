from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from airflow.operators.bash import BashOperator


test_var = "A" # Variable.get("SECRET_TEST_VAR")


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='30 * * * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = GKEStartPodOperator(
        location="europe-west1-b",
        cluster_name="hexa-aldebaran",
        project_id="blsq-dip-test",
        namespace="hexa-airflow",
        name="chirps-download-pod",
        image="blsq/chirps-extraction",
        arguments="download --start 2021 --end 2021 --output-dir /tmp/chirps/".split(" "),
        task_id="chirps-download",
        env_vars={"TEST_VAR": test_var},
    )
    extract = BashOperator(task_id='chirps-extract', bash_command="echo docker run extract")

    download >> extract
