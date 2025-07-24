# ! This script should be called from the parent folder of the repository
# run tests with coverage check
coverage run --omit=./spkanon_eval/NISQA/*,./spkanon_eval/spkanon_eval/featex/wavlm/modules.py,./spkanon_eval/spkanon_eval/featex/wavlm/wavlm_model.py -m unittest discover -s spkanon_eval/tests -p "test_*.py"
coverage combine -m
coverage report -m --include="spkanon_eval/*"
