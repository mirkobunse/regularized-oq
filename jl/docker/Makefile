TAG_NAME=$(USER)/ecml22
USER=$(shell id --user --name)
UID=$(shell id --user)
GROUP=$(shell id --group --name)
GID=$(shell id --group)

push: .PUSH
.PUSH: .IMAGE
	@ echo "The image '${TAG_NAME}' is built for the Docker image repository '${DOCKER_REPOSITORY}'."
	@ echo "You can set the name of this repository with the environment variable DOCKER_REPOSITORY.\n"
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker tag $(TAG_NAME) $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker push $(DOCKER_REPOSITORY)/$(TAG_NAME)
	docker pull $(DOCKER_REPOSITORY)/$(TAG_NAME)
	echo "$(DOCKER_REPOSITORY)/$(TAG_NAME)" > $@

image: .IMAGE
.IMAGE: Dockerfile .Project.toml .Manifest.toml setup.jl startup.jl
	- docker rmi -f $(TAG_NAME)
	docker build \
	    --build-arg user=$(USER) \
	    --build-arg uid=$(UID) \
	    --build-arg group=$(GROUP) \
	    --build-arg gid=$(GID) \
	    --tag $(TAG_NAME) \
	    .
	echo "$(TAG_NAME)" > $@

.Project.toml: ../Project.toml
	grep -A10000 "\[deps\]" $< > $@
.Manifest.toml: ../Manifest.toml
	cp $< $@

clean:
	- docker rmi -f $(TAG_NAME)
	- docker rmi -f $(DOCKER_REPOSITORY)/$(TAG_NAME)
	rm -f .PUSH .IMAGE .Project.toml .Manifest.toml

.PHONY: push image clean
