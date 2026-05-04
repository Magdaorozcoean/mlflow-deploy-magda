.PHONY: train validate pipeline

train:
	python train.py

validate:
	python validate.py

pipeline: train validate