FROM julia:1.0.5-buster
COPY . /src
WORKDIR /src

# put a link to julia in the path
RUN ln -s /usr/local/julia/bin/julia /usr/bin/julia

# set the home directory for config files
ENV HOME=/home

# install required debian packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-cairo python3-venv \
        cmake build-essential pkg-config

# install required python packages
RUN pip3 install -r requirements.txt

# install required julia packages
RUN mkdir -p /home/.julia/environments/v1.0
COPY Project.toml /home/.julia/environments/v1.0/Project.toml
COPY Manifest.toml /home/.julia/environments/v1.0/Manifest.toml
RUN julia -e "import Pkg; Pkg.instantiate()"

# prime the julia package cache
RUN bash -c "for f in *.jl; do ./\$f; done; true"

# make key directories world read/writable to support non-root users
RUN chmod -R a+rw /home && \
    chmod -R a+rw /src

ENTRYPOINT ["/bin/bash"]
