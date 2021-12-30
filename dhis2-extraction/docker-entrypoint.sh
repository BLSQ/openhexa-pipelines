#!/bin/bash
set -e

command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  dhis2extract     : start dhis2extraction subsystem
  python           : run arbitrary python code
  bash             : launch bash session
  test             : launch tests using Pytest

  Any arguments passed will be forwarded to the executed command
  """
}

case "$command" in
"dhis2extraction")
  python -m dhis2extract $arguments
  ;;
"test")
  pytest -s $arguments
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
