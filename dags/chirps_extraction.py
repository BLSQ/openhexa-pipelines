from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator


test_var = Variable.get("SECRET_TEST_VAR")


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='30 * * * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = DockerOperator(
        image="blsq/chirps-extraction",
        docker_url='tcp://localhost:2375',
        command="download --start 2021 --end 2021 --output-dir /tmp/chirps/",
        task_id='chirps-download',
        environment={"TEST_VAR": test_var},
    )
    extract = DockerOperator(
        image="blsq/chirps-extraction",
        docker_url='tcp://localhost:2375',
        command="extract --start 2021 --end 2021",
        task_id='chirps-extract',
        environment={"TEST_VAR": test_var},
    )

    download >> extract
