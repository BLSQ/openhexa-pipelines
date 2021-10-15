OpenHexa Pipelines
==================

This repository contains the self-contained data pipelines developed by BlueSquare for OpenHexa.

Local development
-----------------

### Code style

Our python code is linted using [`black`](https://github.com/psf/black), [`isort`](https://github.com/PyCQA/isort) and 
[`autoflake`](https://github.com/myint/autoflake).
We currently target the Python 3.9 syntax.

We use a [pre-commit](https://pre-commit.com/) hook to lint the code before committing. Make sure that `pre-commit` is
installed, and run `pre-commit install` the first time you check out the code. Linting will again be checked
when submitting a pull request.

You can run the lint tools manually using `pre-commit run --all`.