#!/bin/bash

helm repo add maia https://kthcloud.github.io/MAIA/
helm repo update
python -m pip install maia-toolkit==1.1.2

python manage.py makemigrations authentication
python manage.py makemigrations
python manage.py migrate

cp -r /etc/MAIA-Dashboard/configs/* $CONFIG_PATH
exec "$@"