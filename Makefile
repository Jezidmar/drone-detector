.PHONY: env test get_ckpts prep_split patch_model_dict clean

SRC=EAT-base_epoch30_pt.pt
DST=EAT-base_epoch30_pt_adapted.pt
ENV=drone-detector
DOCKER_IMG=drone-det

env:
	conda create -y -n ${ENV} python=3.13
	conda run -n ${ENV} pip install -r requirements.txt


test:
	python3 -m  pytest -q


get_ckpts:
	gdown https://drive.google.com/uc?id=1PEgriRvHsqrtLzlA478VemX7Q0ZGl889
	gdown https://drive.google.com/uc?id=19hfzLgHCkyqTOYmHt8dqVa9nm-weBq4f


prep_split:
	python3 data/prep_data.py  --output_dir geronimo_processed

patch_model_dict:
	python3 model/adapt_EAT_model_dict.py --src ${SRC}   --dst ${DST}


docker_build:
	docker build -t ${DOCKER_IMG} docker/

docker_run:
	docker run --rm -it ${DOCKER_IMG}


clean:
	rm -rf build dist .pytest_cache .coverage *.egg-info
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -
