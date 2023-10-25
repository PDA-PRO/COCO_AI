## **1\. 소개**

### 1) 소개

[COCO : COdingCOach | 코딩 초보자들을 위한 온라인 저지 | ~ing](https://sjw-record.tistory.com/16)

알고리즘 문제의 오답 코드에서 틀린 부분을 찾아주는 기능을 구현하기 위해 AI를 활용했습니다.

이 기능을 WPC(Wrong Part of code)라고 명명합니다.

### 2) 중요성과 활용 가능성

알고리즘 문제 풀이 중 자신이 작성한 코드 내에서 논리적으로 잘못된 곳으로 인해 오답이 되었을 때 스스로 제출 코드 내에서 어떤 부분이 잘못된 것인지 찾고 고치지 못하는 것을 방지하기 위해 이 모델을 개발하게 되었습니다.

해당 모델에서는 쉬운 수준의 알고리즘 문제에 대한 제출 코드의 논리적 오류를 찾는 것으로 범위를 제한했지만 데이터셋이 더 커진다면 더 다양한 알고리즘 문제에 대해 활용할 수 있을 것이라고 생각됩니다.

---

## **2\. 문제 정의와 목표 설정**

### 1) 문제 정의

알고리즘 문제에 대한 오답 코드에서 문법오류, 런타임 오류에 비해 논리적 오류는 알고리즘 문제별로 발생하는 이유가 다르기 때문에 사람이 직접 코드를 디버깅하여 오류 위치를 찾아 고쳐야합니다. 하지만 코딩을 자주 접하지 못한 초보자들에게는 자신의 코드를 디버깅하고 논리적 오류 위치를 찾는 것은 상당히 어려운 일입니다.

자신의 코드의 오류 위치를 찾지 못하는 일이 반복된다면 코딩에 흥미를 잃게 될 수도 있습니다.

### 2) 목표 설정

따라서, 위 문제를 해결하기 위해 알고리즘 문제와 오답코드를 입력값으로 받아 오답 코드의 논리적 오류를 고친 코드를 출력값으로 하는 AI 모델을 만들기로 했습니다.우선 모델 입력값의 범위를 초보자가 풀만한 쉬운 수준의 문제로만 한정하였습니다.

---

## **3\. AI 기술 선택 및 활용 방법**

### 1) 모델 선택

CodeBERT - 6개 프로그래밍 언어(Python, Java, JavaScript, PHP, Ruby, Go)의 NL-PL 쌍에 대해 사전 훈련된 다중 프로그래밍 언어 모델

GraphCodeBERT - CodeBERT가 소스 코드의 semantic-level structure에 대한 고려 없이 소스 코드를 token들의 시퀸스로만 표현하는 것을 개선하고자 Data Flow를 이용해 코드의 고유 구조(inherent structure)를 고려한 모델

[https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement)

의 Seq2Seq 모델을 약간 수정하여 미세조정

### 2) 모델 구조

![wpc 모델 구조 drawio](https://github.com/PDA-PRO/COCO_AI/assets/80380576/cf923eac-dcf8-4a49-b3e1-9052bb4a3bbb)

왼쪽이 기존 GraphCodeBERT의 refinement task model구조입니다.

오른쪽이 입력값을 수정하여 학습한 모델입니다. 알고리즘 문제의 설명문을 입력값으로 추가하여 각 알고리즘 문제에 대한 논리적 오류들을 학습하도록 하였습니다.

![wpc 모델 구조 drawio](https://github.com/PDA-PRO/COCO_AI/assets/80380576/b876afa3-d156-4813-af6b-2cf6274afbbc)

최종적으로 수정이 완료된 모델의 전체 구조입니다.

---

## **4\. 데이터 수집과 전처리**

### 1) 데이터 수집

\- algorithm problem description

[https://developer.ibm.com/exchanges/data/all/project-codenet/](https://developer.ibm.com/exchanges/data/all/project-codenet/)

IBM Research에서 만든 프로그래밍과 소프트웨어 개발에 관련된 대규모 데이터셋, 다양한 알고리즘 문제와 각 문제에 제출된 다양한 언어의 코드로 이루어져있다.

\- bug code - fixed code 쌍 데이터

FixEval: Execution-based Evaluation of Program Fixes for Competitive Programming Problems 의 데이터 셋을 활용  
codenet의 데이터셋중에서 파이썬으로 작성한 코드만 뽑아내고 한 유저가 한가지 문제를 풀어내기까지의 코드 변화내역을 1:m으로 엮은 데이터셋

### 2) 데이터 전처리

학습 데이터는 크게

1.  FixEval 파이썬 데이터셋에서 논리적 오류가 존재하여 TC를 통과하지 못한 WA코드쌍만 추출
2.  초보자에게 쉬운, 복잡하지 않는 문제를 선별하기 위해 score가 400이상인 문제 삭제
3.  문제별로 같은 의미지만 다른 유니코드의 기호들을 변환
4.  코드쌍 추상화 - 변수와 함수의 이름을 var1 ... varX, method1 ... methodX 로 추상화
5.  코드쌍과 알고리즘 문제 설명문을 토큰화 했을 때 최대 토큰 512를 초과하는 코드쌍 삭제
6.  nl + pl 코드쌍 생성

으로 진행된다.

[https://github.com/PDA-PRO/COCO_AI/blob/main/wpc-finetuning/data_pre_process.py](https://github.com/PDA-PRO/COCO_AI/blob/main/wpc-finetuning/data_pre_process.py)

자세한 전처리 코드는 위 링크를 참고바랍니다.

https://colab.research.google.com/drive/1BBkHdZIsGjUOTp34IGECnWXZywopkHTy?usp=sharing  
위 링크 및 [coco-ai데이터생성](https://github.com/PDA-PRO/COCO_AI/blob/main/wpc-finetuning/coco-ai_데이터생성.ipynb) 를 참고하여 학습 데이터셋을 생성할 수 있습니다.

---

## **5\. 모델 훈련**

### 0) 모델 훈련 해보기

[graphcodebert\_학습](https://github.com/PDA-PRO/COCO_AI/blob/main/wpc-finetuning/graphcodebert_학습.ipynb) 를 참고하여 학습을 진행할 수 있습니다.

### 1) 모델 훈련

```
!python ./COCO_AI/graphcodebert_0718/run.py \
--do_train \
--do_eval \
--model_type roberta \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--train_filename /content/drive/MyDrive/coco_ai/train.json \
--dev_filename /content/drive/MyDrive/coco_ai/val.json \
--p_desc_path /content/drive/MyDrive/coco_ai/tmp_p_desc \
--output_dir /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output \
--load_model_path /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output/checkpoint-last/pytorch_model.bin \
--max_source_length 512 \
--max_target_length 256 \
--beam_size 10 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 5e-5 \
--num_train_epochs 20 2>&1| tee /content/drive/MyDrive/coco_ai/graphcodebert_256_0724/output/train.log
```

### 2) 모델 훈련 과정

Google Colab A100 고용량 RAM 을 사용하여 10epoch 씩 총 20epoch 학습 수행

더보기

.......

\*\*\*\*\* Running evaluation \*\*\*\*\*  
09/10/2023 07:43:18 - INFO - \_\_main\_\_ -     Num examples = 25381  
09/10/2023 07:43:18 - INFO - \_\_main\_\_ -     Batch size = 32  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     eval_ppl = 1.1595  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     global_step = 55981  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     train_loss = 0.029  
09/10/2023 07:47:45 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     bleu-4 = 73.64   
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     xMatch = 37.0   
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     Best BLEU+xMatch:110.64  
09/10/2023 08:08:31 - INFO - \_\_main\_\_ -     \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  
epoch 10 loss 0.0338: 100%|██████████| 5598/5598 \[1:38:14<00:00,  1.05s/it\]

\*\*\*\*\* Running evaluation \*\*\*\*\*

09/10/2023 09:46:47 - INFO - \_\_main\_\_ - Num examples = 25381

09/10/2023 09:46:47 - INFO - \_\_main\_\_ - Batch size = 32

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - eval_ppl = 1.15586

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - global_step = 61579

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - train_loss = 0.0338

09/10/2023 09:51:14 - INFO - \_\_main\_\_ - \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - bleu-4 = 73.29

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - xMatch = 36.8

09/10/2023 10:11:48 - INFO - \_\_main\_\_ - \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*

epoch 11 loss 0.0305: 80%|███████▉ | 4470/5598 \[1:18:27<19:47, 1.05s/it\]

---

## **6. 결과 분석 및 평가**

### 1) 평가 지표

code refinement영역에서 평가지표는 BLEU, CodeBLEU, Acc, TestCase-base평가 등 여러가지가 존재하지만 정확한 평가지표가 없기 때문에 가장 많은 평가지표로 활용된 BLEU, CodeBLEU, Acc를 활용

BLEU-4 : 73.64

xMatch : 37.0

### 2) 테스트 데이터 모델 성능 평가

문제 설명과 오답코드를 입력값으로 넣고 나온 출력코드를 해당 알고리즘 문제의 TC로 채점하여 맞은 비율을 측정

---

## **8\. 참고문헌**

GraphCodeBERT paper
[https://arxiv.org/pdf/2009.08366.pdf](https://arxiv.org/pdf/2009.08366.pdf)

GraphCodeBERT github
[https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/refinement)

CodeNet DataSet paper
[https://arxiv.org/pdf/1812.06469.pdf](https://arxiv.org/pdf/1812.06469.pdf)

CodeNet DataSet github
[https://github.com/IBM/Project_CodeNet](https://github.com/IBM/Project_CodeNet)

refinment paper
[https://arxiv.org/abs/1812.08693](https://arxiv.org/abs/1812.08693)

FixEval dataset github
[https://github.com/mahimanzum/FixEval](https://github.com/mahimanzum/FixEval)
