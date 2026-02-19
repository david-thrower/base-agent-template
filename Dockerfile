FROM python:3.12

WORKDIR /opt
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

# RUN python 0_load_db.py

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
