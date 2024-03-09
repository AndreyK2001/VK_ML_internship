# FROM python:3.8.6-buster

# FROM nvidia/cuda:10.2-devel-ubuntu18.04

FROM nvidia/cuda:11.0.3-base-ubuntu18.04

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils wget && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app
ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

#RUN chmod +x $PROJECT_ROOT/lib/load_fine_tuned.sh && \
#    $PROJECT_ROOT/lib/load_fine_tuned.sh

RUN pip install -r requirements.txt

RUN cd $PROJECT_ROOT/lib/

RUN wget "https://getfile.dokpub.com/yandex/get/https://disk.yandex.com/d/UHwmj9UQyDcC6g" -O tuned_model_weights

RUN python load_model_weights.py

CMD ["python", "lib/run.py"]
