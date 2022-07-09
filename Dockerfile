FROM python:3.10

COPY . .

RUN apt install ffmpeg -y
RUN /usr/local/bin/python3 -m pip install --upgrade pip setuptools
RUN /usr/local/bin/python3 -m pip install -r requirements.txt

EXPOSE 8501
CMD [ "streamlit", "run", "main.py" ]