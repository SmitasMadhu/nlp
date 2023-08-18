default: all

install:
	pip3 install -r requirements/test_requirements.txt
		
train:
	python ./sentimental_model/train_model.py 2>error.log

test:
	python -m pytest tests/test_*.py 2>error.log

build:
	python -m build  2> error.log

dockerize:
	cp dist/*.whl sentimental_model
	docker build ${PWD} -t sentimental_model:v1

all : install train test 
