FROM python:3.8-slim-buster
ENV PATH="/scripts:${PATH}"
COPY ./requirements.txt ./requirements.txt
RUN apt-get update -y
RUN apt-get install gcc -y
RUN pip install torch==1.11.0 --no-cache
RUN pip install -r /requirements.txt --no-cache
RUN mkdir /arq
COPY ./arq /arq
WORKDIR /arq
COPY ./scripts /scripts
RUN chmod +x /scripts/*
RUN mkdir -p /vol/web/media
RUN mkdir -p /vol/web/static

RUN adduser --disabled-password user
RUN chown -R user:user /vol
RUN chown -R user:user /arq
RUN chmod -R 755 /vol/web
RUN chown -R user:user celerybeat-schedule.dat
RUN chown -R user:user celerybeat-schedule.bak
RUN chown -R user:user celerybeat-schedule.dir
USER user


CMD ["entrypoint.sh"]
