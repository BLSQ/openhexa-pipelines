from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator


test_var = Variable.get("SECRET_TEST_VAR")


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='30 * * * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = BashOperator(task_id='chirps-download', bash_command="echo docker run download $TEST_VAR", env={"TEST_VAR": test_var})
    extract = BashOperator(task_id='chirps-extract', bash_command="echo docker run extract")

    download >> extract
