from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.bash import BashOperator


test_var = "A" # Variable.get("SECRET_TEST_VAR")


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='30 * * * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = KubernetesPodOperator(
        namespace='hexa-airflow',
        image="blsq/chirps-extraction",
        arguments="download --start 2021 --end 2021 --output-dir /tmp/chirps/".split(" "),
        name="chirps-download-pod",
        task_id="chirps-download",
        env_vars={"TEST_VAR": test_var},
        in_cluster=True,
        get_logs=True,
    )
    extract = BashOperator(task_id='chirps-extract', bash_command="echo docker run extract")

    download >> extract
