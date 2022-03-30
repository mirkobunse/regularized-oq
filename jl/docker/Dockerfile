FROM julia:1.6

# install all tools and create the user
ARG user
ARG uid
ARG group
ARG gid
ENV PYTHON=""
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        gettext \
        git \
        hdf5-tools \
        imagemagick \
        less \
        libgtk-3-0 \
        make \
        nano \
        procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid $gid $group && \
    useradd --gid $gid --uid $uid --create-home $user && \
    echo "alias ll='ls -halF'" >> /root/.bashrc && \
    echo "alias ll='ls -halF'" >> /home/$user/.bashrc

# install all Julia packages
ADD setup.jl startup.jl /home/$user/.julia/config/
ADD .Project.toml /home/$user/.julia/environments/v1.6/Project.toml
ADD .Manifest.toml /home/$user/.julia/environments/v1.6/Manifest.toml
RUN chown --recursive $user:$gid /home/$user/
USER $uid
RUN julia --optimize=3 --startup-file=no /home/$user/.julia/config/setup.jl

WORKDIR /mnt/home/deconv-exp
CMD ["bash"]
