# COCO_Back_End
CBNU SW Seniar Project COCO - Fast API

## Getting started
### Environmental preparation (Windows)

+ System: Windows 10

1. Install Docker

https://docs.docker.com/desktop/install/windows-install/

Since it is divided into an installation method using WSL2 and an installation method using Hyper-V, installation using the appropriate installation method is required.

### Install

#### Build and install Docker image

1. Please select a location with some surplus disk space and run the following command:

    ```bash
    git clone https://github.com/PDA-PRO/COCO_Back_End.git && cd COCO_Back_End
    ```

2. Start service

    ```bash
    docker-compose up -d --build
    ```

## Environmental preparation (Linux)

+ System: Ubuntu 20.04.6 LTS

1. Install Docker

    ```bash
    sudo apt update
    sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io
    sudo usermod -aG docker
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    ```

2. Install Docker-Compose

    ```bash
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    ```

    Other installation methods: [https://docs.docker.com/install/](https://docs.docker.com/install/)

## Install

1. Please select a location with some surplus disk space and run the following command:

    ```bash
    git clone https://github.com/PDA-PRO/COCO_Back_End.git && cd COCO_Back_End
    ```

2. Start service

    ```bash
    docker-compose up -d --build
    ```
