FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    vim \
    sudo \
    curl \
    apt-utils \
    openssh-server \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
USER root

# Docker containers max sys uid is 999 but slurm defaults to a larger one
RUN groupadd -r slurm && useradd -r -g slurm -u 999 slurm

RUN echo "ubuntu ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

RUN sed -i 's/#PermitRootLogin/PermitRootLogin/' /etc/ssh/sshd_config && \
    mkdir -p /home/ubuntu/.ssh && \
    touch /home/ubuntu/.ssh/authorized_keys && \
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh /home/ubuntu/.ssh/authorized_keys && \
    chmod 700 /home/ubuntu/.ssh && \
    chmod 600 /home/ubuntu/.ssh/authorized_keys

#   22: SSH
# 6818: Slurm D
EXPOSE 22 6818