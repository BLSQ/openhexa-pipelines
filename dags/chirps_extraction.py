from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='30 * * * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = BashOperator(task_id='chirps-download', bash_command="echo docker run download")
    extract = BashOperator(task_id='chirps-extract', bash_command="echo docker run extract")

    download >> extract
