#!/bin/bash
set -e

if [ ! -f "bin/gen_model" ]; then
    ./build.sh
fi

./bin/gen_model
