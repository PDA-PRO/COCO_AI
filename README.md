# COCO-AI

## 개요

알고리즘 문제의 오답 코드에 대한 논리적 오류를 찾아 고쳐주는 모델  
COCO의 플러그인 WPC(Wrong Part of Code)의 기반 모델을 위함

## 시작하기

### 환경 준비

#### Windows

- System: Windows 10

1. 도커 설치

https://docs.docker.com/desktop/install/windows-install/

WSL2를 이용한 설치방법과 Hyper-V를 이용한 설치방법으로 나누어져 있으므로 그에 맞는 설치방법을 이용한 설치가 필요합니다.

#### Linux

- System: Ubuntu 20.04.6 LTS

1. 도커 설치

   ```bash
   sudo apt update
   sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   sudo usermod -aG docker
   ```

### 설치하기

#### 도커 이미지를 빌드하고 설치하기

1. 저장소 복제하기

   ```bash
   git clone https://github.com/PDA-PRO/COCO_AI.git
   cd COCO_AI/wpc
   ```

2. 미세 조정된 모델 파라미터 다운로드 및 `COCO_AI/wpc`폴더로 이동  
   [pytorch_model.bin](https://drive.google.com/file/d/15hw7W_dWtZituw4-alawS-mdG8X4Vj38/view?usp=sharing)

3. 도커 이미지를 빌드하기

   ```bash
   docker build -t coco-ai
   ```

   네트워크의 상황에 따라 모든 설정이 완료 될 때까지 5~30분의 시간이 소요될 수 있습니다.

4. 컨테이너 실행하기

   모델을 테스트하기 위해 원하는 \<port\>를 지정해주세요

   - gpu로 테스트하기

   ```bash
   docker run --gpus all --name coco-ai -it -p <port>:8000 coco-ai:lastet
   ```

   - gpu없이 cpu로만 테스트하기

   ```bash
   docker run --name coco-ai -it -p <port>:8000 coco-ai:lastet
   ```

#### 도커 이미지를 불러와서 설치하기

1. 도커 이미지를 불러옵니다.

   ```bash
   docker pull styughjvbn/coco-ai:1.0.1
   ```

   네트워크의 상황에 따라 모든 설정이 완료 될 때까지 5~30분의 시간이 소요될 수 있습니다.

2. 컨테이너 실행하기

   모델을 테스트하기 위해 원하는 \<port\>를 지정해주세요

   - gpu로 테스트하기

   ```bash
   docker run --gpus all --name coco-ai -it -p <port>:8000 styughjvbn/coco-ai:1.0.1
   ```

   - gpu없이 cpu로만 테스트하기

   ```bash
   docker run --name coco-ai -it -p <port>:8000 styughjvbn/coco-ai:1.0.1
   ```

### 실행하기

Swagger를 사용하여 모델을 테스트해볼 수 있습니다.  
웹 브라우저에서 http://localhost:<port\>/docs 를 방문하여 테스트해볼 수 있습니다.
![image](https://github.com/PDA-PRO/COCO_Front_End/assets/80380576/ecc2cdc0-63de-4bc3-b0c5-b4c22a39e3e8)

## 학습하기

현재 저희의 모델은 컴퓨터 자원의 부족, 학습 데이터셋의 질과 양의 한계로 좋은 성능을 내지 못합니다. 저희의 [학습 과정](https://github.com/PDA-PRO/COCO_AI/tree/main/wpc-finetuning#readme)을 참고해서 더욱 좋은 모델을 만들어 보세요

## 라이선스
[Apache2.0](https://www.apache.org/licenses/LICENSE-2.0)
