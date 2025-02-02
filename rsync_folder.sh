#!/bin/bash

SERVER=mstirlin@puhti.csc.fi
TARGET=/projappl/project_2012962/datasets/

if [ -z "$1" ]; then
    echo "please pass the path to a folder to rsync"
    exit 1
fi

SOURCE="$1"

# make sure folder exists
if [ ! -d $SOURCE ]; then
    echo "no such folder '$1'"
    exit 1
fi

echo "rsyncing '$1' to '$SERVER:$DIRECTORY'"

# determine if dry run or not
DRY_RUN="--dry-run"
if [ ! -z "$2" ]; then
    if [ "$2" = "--run" ]; then
        DRY_RUN=""
    else
        echo "Error: Unrecognized argument '$2', use '--run'"
    fi
fi
echo

# rsync command
rsync -aPvh $DRY_RUN \
    --bwlimit=1000 \
    --delete \
    --progress \
    "$SOURCE" "$SERVER:$TARGET"

# final echos
echo
if [[ -z "$2" || $2 != "--run" ]]; then
    echo "This was just for FUNZIES! To copy for REALZIES, use '--run'"
elif [ $2 != "--run" ]; then
    echo "Unrecognized argument '$2'"
fi
echo
