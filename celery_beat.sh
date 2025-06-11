#!/bin/bash

NAME="celery-beat"  #Django application name
DIR=/home/ubuntu/deeplobe-api   #Directory where project is located
USER=ubuntu   #User to run this script as
GROUP=ubuntu  #Number of workers that Gunicorn should spawn 
DJANGO_SETTINGS_MODULE=deeplobe_api.settings.production 
cd $DIR
source venv/bin/activate  #Activate the virtual environment
source .env
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DIR:$PYTHONPATH


#Command to run the progam under supervisor
exec  celery -A deeplobe_api  beat -l INFO 