#!/bin/bash

helm repo add maia https://kthcloud.github.io/MAIA/
helm repo update
python -m pip install maia-toolkit

python manage.py makemigrations authentication
python manage.py makemigrations
python manage.py migrate

exec "$@"