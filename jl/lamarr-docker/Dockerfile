FROM julia:1.6

MAINTAINER Mirko Bunse <mirko.bunse@cs.tu-dortmund.de>

ARG USER

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux
ENV PYTHON=""

RUN apt-get update && apt-get install -qy --no-install-recommends\
	gettext \
	hdf5-tools \
	imagemagick \
	libgtk-3-0 \
	make \
	procps \
	openssh-server\
	git\
	vim\
	git\
	htop\
	less\
	zip\
	rsync\
	tmux\
	screen\
	emacs\
	nano\
	build-essential\
	curl\
	wget\
	rsync\
	unzip\
	iputils-ping\
	locales locales-all &&\
        locale-gen en_US.UTF-8 && locale-gen de_DE.UTF-8

RUN echo "SFB876" > /etc/defaultdomain

COPY ./resource/container /container
RUN chmod +x /container/startup.sh

COPY ./resource/ssh/ssh_host* /etc/ssh/
RUN chmod 600 /etc/ssh/ssh_host* && chmod 644 /etc/ssh/ssh_host*.pub

RUN echo 'echo ###############################################' >> /etc/bash.bashrc; echo 'echo ################ 1.0 #########################' >> /etc/bash.bashrc; echo 'echo PORT NUMBER: ${SSH_PORT}' >> /etc/bash.bashrc; echo 'echo HOST: ${SLURMD_NODENAME}' >> /etc/bash.bashrc; echo 'echo To connect via an SSH client to this job use ssh ${SLURMD_NODENAME}s0 -p ${SSH_PORT}' >> /etc/bash.bashrc; echo 'echo Do not close this session, otherwise the job will be terminated!' >> /etc/bash.bashrc; echo 'echo ###############################' >> /etc/bash.bashrc;

# Change to non-root privilege

ENTRYPOINT ["/container/startup.sh"]
