#!/bin/bash

helm repo add maia https://kthcloud.github.io/MAIA/
helm repo update
#python -m pip install maia-toolkit
git clone https://github.com/kthcloud/MAIA.git
pip install ./MAIA
python manage.py makemigrations authentication
python manage.py makemigrations gpu_scheduler
python manage.py makemigrations
python manage.py migrate

exec "$@"