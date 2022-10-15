# 월간 데이콘 심리 성향 예측 AI 경진대회
>설문조사를 통해 얻은 tabular 데이터셋을 사용해 국가 선거 투표여부를 예측하는 대회

</br>

## 1. 제작 기간 & 참여 인원
- 2020년 09월 28일 ~ 2020년 11월 16일
- 개인으로 참여

</br>

## 2. 사용 기술
- python
- pytorch
- scikit-learn
- DNN

</br>

## 3. file 설명
`deeplearning_pytorch.py` data preprocessing, training model, prediction

`model.py` DNN

`util.py` 표 데이터의 각 컬럼별 pipeline

</br>

## 4. 트러블 슈팅
### 컬럼별 전처리
- 표 데이터의 컬럼별 특성이 달라 같은 파이프라인을 적용할 수 없었다.
- 사이킷런을 이용하여 각 컬럼별 transformer를 만들고 columntransformer을 사용해 하나의 파이프라인으로 만들었다.
