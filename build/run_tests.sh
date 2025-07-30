#! /bin/bash
# Runs tests with coverage check
# This script should be called from the parent folder of the repository
coverage run --omit=./spane/NISQA/*,./spane/spkanon_eval/featex/wavlm/modules.py,./spane/spkanon_eval/featex/wavlm/wavlm_model.py -m unittest discover -s spane/tests -p "test_*.py"

if [ $? -eq 1 ]; then
    message="Unit tests failed, please check and fix your code."
    echo -e "\033[1;31mERROR: $message\033[0m";
    exit 1
else
    message="Passed unit tests."
    echo -e "\033[1;32mOK: $message\033[0m"
    coverage combine
    coverage report -m --include="spane/*"
    exit 0
fi
