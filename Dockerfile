FROM python:3.12

WORKDIR /opt
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7680

ENTRYPOINT ["python", "2_run_agent.py"]
