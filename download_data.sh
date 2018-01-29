#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

export PYTHONPATH=.

wget https://people.eng.unimelb.edu.au/tbaldwin/resources/satire/satire.tgz
tar -xvf satire.tgz
rm satire.tgz
