default: pylint pytest

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes


# .DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y dfake || :
	@pip install -e .

run_train:
	python -c 'from dfake.interface.main import train; train()'

run_pred:
	python -c 'from dfake.interface.main import pred; pred()'

run_evaluate:
	python -c 'from dfake.interface.main import evaluate; evaluate()'

run_all: run_train run_pred run_evaluate


##################### TESTS #####################

# test_gcp_setup:
# 	@pytest \
# 	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
# 	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
# 	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
# 	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_project_id \
# 	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_bucket_name

# default:
# 	cat tests/lifecycle/test_output.txt

# test_mlflow_config:
# 	@pytest \
# 	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_experiment_is_not_null \
# 	tests/lifecycle/test_mlflow.py::TestMlflow::test_mlflow_model_name_is_not_null
