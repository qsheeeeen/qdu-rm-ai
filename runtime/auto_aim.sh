#!/bin/bash

if [ ! -d "../runtime/" ]; then
  echo "Script must run from 'runtime'."
  exit
else
  echo "Start auto-aim"
fi

auto-aim
