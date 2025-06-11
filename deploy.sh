#!/bin/bash

NAME="deeplobe"  #Django application name
WORKERS=17    #Number of workers that Gunicorn should spawn 
DJANGO_SETTINGS_MODULE=deeplobe_api.settings.production   #Which Django setting file should use
DJANGO_ASGI_MODULE=deeplobe_api.wsgi           #Which WSGI file should use
LOG_LEVEL=debug
TIMEOUT=360


export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE


source .env
#Command to run the progam under supervisor
exec gunicorn --bind 0.0.0.0:8000 ${DJANGO_ASGI_MODULE}:application \
--workers $WORKERS \
--timeout $TIMEOUT \
--log-level $LOG_LEVEL