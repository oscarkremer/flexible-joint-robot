#!/bin/bash

PATH=$PATH:/home/oscarkremer/miniconda3/bin
cd /home/oscarkremer/cerberus
source activate
conda activate cerberus
make webserver gunicorn app:app -b 0.0.0.0:8000