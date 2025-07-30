#!/bin/bash
# Script to control SHIRG debug output

if [ "$1" = "on" ]; then
    echo "Enabling SHIRG debug output..."
    export SHIRG_DEBUG=1
    echo "SHIRG_DEBUG=1"
elif [ "$1" = "off" ]; then
    echo "Disabling SHIRG debug output..."
    export SHIRG_DEBUG=0
    echo "SHIRG_DEBUG=0"
else
    echo "Usage: source control_debug.sh [on|off]"
    echo "Current SHIRG_DEBUG=$SHIRG_DEBUG"
fi