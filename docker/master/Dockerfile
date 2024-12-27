FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    sudo \
    curl \
    apt-utils \
    build-essential \
    devscripts \
    debhelper \
    openssh-server \
    openssh-client \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
USER root

# Docker containers max sys uid is 999 but slurm defaults to a larger one
RUN groupadd -r slurm && useradd -r -g slurm -u 999 slurm

RUN echo "ubuntu ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

COPY . /home/ubuntu/dtt/

RUN chown -R ubuntu:ubuntu /home/ubuntu/dtt/

RUN dpkg -i /home/ubuntu/dtt/slurmify_0.1.0_amd64.deb

#   22: SSH
# 6817: Slurm Ctl D
# 6818: Slurm D
EXPOSE 22 6817 6818