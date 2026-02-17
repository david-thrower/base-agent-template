FROM python:3.12

WORKDIR /opt
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python 0_load_db.py

ENTRYPOINT ["python", "2_run_agent.py"]
