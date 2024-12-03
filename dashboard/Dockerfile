FROM kthcloud/maia-workspace-admin:1.5


USER root
# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get install -y default-libmysqlclient-dev build-essential pkg-config
RUN mkdir /etc/MAIA-Dashboard

WORKDIR /etc/MAIA-Dashboard
COPY . .

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#COPY env env


# running migrations
#RUN python manage.py makemigrations
#RUN python manage.py migrate
#RUN python manage.py generate-api

ENTRYPOINT ["bash", "start.sh"]
# gunicorn
CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000","--insecure"]
#CMD ["gunicorn", "--config", "gunicorn-cfg.py", "core.wsgi"]

