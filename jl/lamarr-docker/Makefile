EXPORT_NAME=nvcr.io/ml2r/lamarr-dortmund/interactive_julia:1.6

all: .IMAGE

.IMAGE: custom-container-example/Dockerfile
	- docker rmi ${EXPORT_NAME}
	cd custom-container-example/ && \
	  docker build -f Dockerfile --network=host -t ${EXPORT_NAME} .
	docker push ${EXPORT_NAME}
	echo ${EXPORT_NAME} > $@

custom-container-example/Dockerfile: Dockerfile
	cp $< $@

.PHONY: all
