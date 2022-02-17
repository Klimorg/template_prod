FROM tiangolo/uvicorn-gunicorn:python3.8

RUN apt-get update && apt-get upgrade -y
RUN apt install libgl1-mesa-glx libglib2.0-0 -y

RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]