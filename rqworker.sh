#!/bin/bash

NAME="rqworker"  #Django application name
DJANGO_SETTINGS_MODULE=deeplobe_api.settings.production 
 #Activate the virtual environment
source .env
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE

#Command to run the progam under supervisor
exec python manage.py rqworker