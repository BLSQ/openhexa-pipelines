FROM blsq/openhexa-base-notebook:latest

COPY __init__.py app.py common.py* /home/jovyan/
WORKDIR /home/jovyan/
ENTRYPOINT ["/home/jovyan/app.py"]
CMD ["--help"]
