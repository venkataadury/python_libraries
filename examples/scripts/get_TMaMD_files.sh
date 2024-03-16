#!/bin/bash

# This script will get the necessary file to run TMaMD_PostProcess notebook from examples

# --- User-specific details (Start) --- #
USERNAME="vadury"
SERVER="bridges2.psc.edu"
DATA_SERVER="data.bridges2.psc.edu"
GET_PATH="/jet/home/vadury/r1128/thermomaps/example4"
# --- User-specific details (End) --- #

ssh -t $USERNAME@$SERVER "cd $GET_PATH/data; zip -rT ship.zip iter*/mount/xtc rosetta iter*/mount/multitraj.h5 iter*/mount/figs"
scp $USERNAME@$DATA_SERVER:$GET_PATH/data/ship.zip .
unzip ship.zip
if test $? -eq 0
then
	rm ship.zip
fi
echo "Done"
