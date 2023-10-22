# COCO-AI
CBNU SW Seniar Project COCO - AI

WPC(Wrong Part of Code) Plugin

## Getting started
### Environmental preparation (Windows)

+ System: Windows 10

1. Install Docker

https://docs.docker.com/desktop/install/windows-install/

Since it is divided into an installation method using WSL2 and an installation method using Hyper-V, installation using the appropriate installation method is required.

### Install

#### Pull and install Docker image

1. Run the command below in Powershell to pull the Docker image.

    ```powershell
    docker pull styughjvbn/coco-ai:1.0.0
    ```

2. To install the image, run the following command in Powershell

    **\<port\> Specify port to open swagger for testing wpc**
   - Operates with gpu
    ```powershell
    docker run --gpus all -it -p <port>:8000 styughjvbn/coco-ai:1.0.0
    ```
   - Operates without gpu, only cpu
    ```powershell
    docker run -it -p <port>:8000 styughjvbn/coco-ai:1.0.0
    ```

3. Testing in web environment swagger

    Test operation at http://localhost:<port\>/docs

#### Build and install Docker image

1. Select a location with extra disk space and run the following command in powershell

    ```powershell
    git clone -b wpc https://github.com/PDA-PRO/COCO_AI.git
    cd COCO_AI/wpc
    ```

2. Run the following command in Powershell to build the image

    ```powershell
    docker build -t wpc
    ```
    
3. To install the image, run the following command in Powershell

    **\<port\> Specify port to open swagger for testing wpc**
   - Operates with gpu
    ```powershell
    docker run --gpus all -it -p <port>:8000 wpc:lastet
    ```
   - Operates without gpu, only cpu
    ```powershell
    docker run -it -p <port>:8000 wpc:lastet
    ```

4. Testing in web environment swagger

    Test operation at http://localhost:<port\>/docs

### Environmental preparation (Linux)

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
    ```

### Install


#### Build and install Docker image

1. Please select a location with some surplus disk space and run the following command:

    ```bash
    git clone -b wpc https://github.com/PDA-PRO/COCO_AI.git
    cd COCO_AI/wpc
    ```

2. Run the following command to build the image

    ```powershell
    docker build -t wpc
    ```

3. To install the image, run the following command

    **\<port\> Specify port to open swagger for testing wpc**
   - Operates with gpu
    ```powershell
    docker run --gpus all -it -p <port>:8000 wpc:lastet
    ```
   - Operates without gpu, only cpu
    ```powershell
    docker run -it -p <port>:8000 wpc:lastet
    ```

4. Testing in web environment swagger

    Test operation at http://localhost:<port\>/docs
