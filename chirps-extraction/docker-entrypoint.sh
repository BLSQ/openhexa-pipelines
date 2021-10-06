#!/bin/bash
set -e

command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  chirps           : start chips subsystem
  tests            : generate a django migration
  python           : run arbitrary python code
  bash             : launch bash session

  Any arguments passed will be forwarded to the executed command
  """
}

case "$command" in
"chirps")
  python -m chirps $arguments
  ;;
"test")
  pytest tests/
  ;;
"python")
  python $arguments
  ;;
"bash")
  bash $arguments
  ;;
*)
  show_help
  ;;
esac
