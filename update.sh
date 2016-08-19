#!/bin/bash
echo Updating V2G-Sim...
pip uninstall v2gsim
BASEDIR=$(dirname "$0")
pip install $BASEDIR
echo Done