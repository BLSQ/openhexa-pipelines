#!/bin/bash
set -e

command=$1
arguments=${*:2}

show_help() {
  echo """
  Available commands:

  temperature      : start temperature subsystem
  python           : run arbitrary python code
  bash             : launch bash session

  Any arguments passed will be forwarded to the executed command
  """
}

case "$command" in
"temperature")
  python -m temperature $arguments
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
