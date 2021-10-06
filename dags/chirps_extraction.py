from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.operators.bash import BashOperator


# set credential
envz = {
    "AWS_ACCESS_KEY_ID": Variable.get("AWS_BUCKET_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_BUCKET_ACCESS_SECRET"),
    "AWS_DEFAULT_REGION": "eu-west-2",
}


with DAG('chirps-extraction', description='CHIRPS Extraction', schedule_interval='0 1 27 * *', start_date=datetime(2021, 10, 1), catchup=False) as dag:
    download = KubernetesPodOperator(
        namespace='hexa-airflow',
        image="blsq/chirps-extraction",
        arguments="chirps download --start 2020 --end 2020 --output-dir s3://hexa-demo-pipeline-chirps/download/".split(" "),
        name="chirps-download-pod",
        task_id="chirps-download",
        env_vars=envz,
        in_cluster=True,
        get_logs=True,
    )
    extract = KubernetesPodOperator(
        namespace='hexa-airflow',
        image="blsq/chirps-extraction",
        arguments="chirps extract --start 2020 --end 2020 --input-dir s3://hexa-demo-pipeline-chirps/download/ --output-file s3://hexa-demo-pipeline-chirps/extract.csv --contours s3://hexa-demo-pipeline-chirps/world/BFA.geo.json".split(" "),
        name="chirps-extract-pod",
        task_id="chirps-extract",
        env_vars=envz,
        in_cluster=True,
        get_logs=True,
    )

    download >> extract
