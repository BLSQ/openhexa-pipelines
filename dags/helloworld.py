from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def print_hello():
 return 'Hello Wolrd'

with DAG('hello_world', description='Hello world example', schedule_interval='* * * * *', start_date=datetime(2020, 9, 25), catchup=False) as dag:
  task1 = PythonOperator(task_id='hello_task', python_callable=print_hello)

task1
