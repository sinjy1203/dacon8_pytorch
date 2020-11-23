## 패키지 추가
import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

## x와 y 나누기
class Split_XY(BaseEstimator, TransformerMixin):
    def __init__(self, label):
        self.label = label
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        try:
            train_x = x.drop("voted", axis=1)
            train_y = x["voted"].copy()
        except:
            train_x = x

        return train_y if self.label else train_x

## age group pipeline
class AgeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, seventy_trans=70.0):
        self.seventy_trans = seventy_trans

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x[x == "10s"] = 10.0
        x[x == "20s"] = 20.0
        x[x == "30s"] = 30.0
        x[x == "40s"] = 40.0
        x[x == "50s"] = 50.0
        x[x == "60s"] = 60.0
        x[x == "+70s"] = self.seventy_trans
        x = x.astype(np.float64)

        return x

## education pipeline
class EducationTransform(BaseEstimator, TransformerMixin):
    def __init__(self, other_trans=2):
        self.other_trans = other_trans

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x[x == 0] = 2.0
        x = x.astype(np.float64)

        return x

## one hot(engnat, gender, hand, race, religion, urban, wr_, wf_)
from sklearn.preprocessing import OneHotEncoder

## family size pipeline
class FamilyTransform(BaseEstimator, TransformerMixin):
    def __init__(self, top_clip=50):
        self.train = False
        self.top_clip = top_clip

    def fit(self, x, y=None):
        self.train = True
        return self

    def transform(self, x):
        if self.train:
            x[x > self.top_clip] = self.top_clip
        self.train = False

        return x

## married pipeline
class MarriedTransform(BaseEstimator, TransformerMixin):
    def __init__(self, other_trans=1.5):
        self.other_trans = other_trans

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x[x == 0] = 1.5
        x = x.astype(np.float64)

        return x

## TP  pipeline
class Q_TpTransform(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.astype(np.float64)

        return x

## voted pipeline (label)

class VotedTransform(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x[x == 2] = 0.0

        x = x.astype(np.float64)

        return x

##
class EngnatTransform(BaseEstimator, TransformerMixin):
    def __init__(self, none_ans=1):
        self.none_ans = none_ans
        self.onehot = OneHotEncoder()

    def fit(self, x, y=None):
        x[x == 0] = self.none_ans
        self.onehot.fit(x)
        return self

    def transform(self, x):
        x[x == 0] = self.none_ans
        return self.onehot.transform(x)

##
class HandTransform(BaseEstimator, TransformerMixin):
    def __init__(self, none_ans=1):
        self.none_ans = none_ans
        self.onehot = OneHotEncoder()

    def fit(self, x, y=None):
        x[x == 0] = self.none_ans
        self.onehot.fit(x)
        return self

    def transform(self, x):
        x[x == 0] = self.none_ans
        return self.onehot.transform(x)

## 파이토치 데이터셋
class datasets(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = {'x': self.x[idx], 'y': self.y[idx]}
        return data