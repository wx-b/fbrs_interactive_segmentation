FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get upgrade
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/MACILLAS/fbrs_interactive_segmentation.git

COPY run.py ./fbrs_interactive_segmentation

WORKDIR /usr/src/app/fbrs_interactive_segmentation

COPY . .

CMD ["python", "./run.py"]