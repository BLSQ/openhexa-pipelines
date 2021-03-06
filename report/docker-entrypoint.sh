#!/bin/bash
set -e

all=("$@")
command="${all[0]}"
arguments=("${all[@]:1}")

show_help() {
  echo """
  Available commands:

  report           : start report subsystem
  python           : run arbitrary python code
  bash             : launch bash session
  test             : launch tests using Pytest

  Any arguments passed will be forwarded to the executed command
  """
}

case $command in
"report")
  python -m report "${arguments[@]}"
  ;;
"test")
  pytest -s "${arguments[@]}"
  ;;
"python")
  python "${arguments[@]}"
  ;;
"bash")
  bash "${arguments[@]}"
  ;;
*)
  show_help
  ;;
esac
