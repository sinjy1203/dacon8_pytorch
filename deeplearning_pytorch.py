## 압축풀기
import os
import shutil
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.functional import F

from model import *
from util import *

data_zip_n = "data.zip"
datasets_dir = "./datasets"
# datasets_dir = "/content/drive/My Drive/Colab Notebooks/training_dacon8/datasets"

## 데이터 로드
data_dir = os.path.join(datasets_dir, "data")
train_dir = os.path.join(data_dir, "train.csv")
test_dir = os.path.join(data_dir, "test_x.csv")
sub_dir = os.path.join(data_dir, "sample_submission.csv")

train = pd.read_csv(train_dir)

## result dir
result_dir = os.path.join(data_dir, "result")
os.makedirs(result_dir, exist_ok=True)

## 데이터 체크
train.describe()

## 테스트 세트 만들기
train = train.drop("index", axis=1)
# train_set = train
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)

## 상관관계 조사
corr_matrix = train.corr()
corr_matrix["voted"].sort_values()

## x와 y 나누기
train_x = train_set.drop("voted", axis=1)
train_y = train_set["voted"].copy()
val_x = val_set.drop('voted', axis=1)
val_y = val_set['voted'].copy()

## 최종 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

## x 파이프라인
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

train_x_index = list(train_x)

q_tp_index = list(filter(lambda x : True if x.startswith('Q') or x.startswith('t') else False, train_x_index))
onehot_index = list(filter(lambda x: True if x == "engnat" or x == "gender" or x == "hand" or x == "race"
                                            or x == "religion" or x == "urban" or x.startswith("w") else False,
                            train_x_index))

attribs_pipeline = ColumnTransformer([("q_tp", Q_TpTransform(), q_tp_index),
                                      ("onehot", OneHotEncoder(), onehot_index),
                                      ("age_group", AgeTransform(seventy_trans=75.5), ["age_group"]),
                                      ("education", EducationTransform(other_trans=2.5), ["education"]),
                                      ("married", MarriedTransform(other_trans=1.5), ["married"]),
                                      ("familysize", FamilyTransform(top_clip=35), ['familysize'])
                                      ])
x_pipeline = Pipeline([("attribs", attribs_pipeline), ("std_scaler", StandardScaler())])

train_x_transformed = np.array(x_pipeline.fit_transform(train_x))
val_x_transformed = np.array(x_pipeline.transform(val_x))

## y 파이프라인
y_pipeline = Pipeline([("voted", VotedTransform())])

train_y_transformed = np.array(y_pipeline.fit_transform(train_y))
val_y_transformed = np.array(y_pipeline.transform(val_y))

## 트레이닝

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
net = Net().to(device)

fn_loss = nn.BCELoss().to(device)
fn_pred = lambda x : (x > 0.5).astype(np.int)
optim = torch.optim.Adam(net.parameters(), lr=0.1)

train_datasets = datasets(x=train_x_transformed, y=train_y_transformed)
train_dataloader = DataLoader(train_datasets, batch_size=64, shuffle=True)

val_datasets = datasets(x=val_x_transformed, y=val_y_transformed)
val_dataloader = DataLoader(val_datasets, batch_size=64)

for epoch in range(10):
    net.train()

    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    for data in train_dataloader:
        x, y = torch.tensor(data['x'], dtype=torch.float32).to(device), \
               torch.tensor(data['y'], dtype=torch.float32).to(device)[:, np.newaxis]

        output = net(x)
        loss = fn_loss(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss_arr += [loss.item()]
        train_acc_arr += [np.mean((fn_pred(output.detach().cpu().numpy()) == y.cpu().numpy()).astype(np.int))]

    with torch.no_grad():
        net.eval()

        for data in val_dataloader:
            x, y = torch.tensor(data['x'], dtype=torch.float32).to(device), \
                   torch.tensor(data['y'], dtype=torch.float32).to(device)[:, np.newaxis]

            output = net(x)
            loss = fn_loss(output, y)

            val_loss_arr += [loss.item()]
            val_acc_arr += [np.mean((fn_pred(output.detach().cpu().numpy()) == y.cpu().numpy()).astype(np.int))]

    train_loss_arr = np.array(train_loss_arr)
    train_acc_arr = np.array(train_acc_arr)
    val_loss_arr = np.array(val_loss_arr)
    val_acc_arr = np.array(val_acc_arr)
    print("EPOCH: %04d | LOSS: %.3f | ACC: %.3f || VAL  LOSS: %.3f | ACC: %.3f" %
          (epoch, np.mean(train_loss_arr), np.mean(train_acc_arr), np.mean(val_loss_arr), np.mean(val_acc_arr)))

## 정확도 확인
# from sklearn.model_selection import cross_val_score
# # scores = cross_val_score(svm_clf, train_x_transformed, train_y_transformed, cv=5, scoring="accuracy")
# scores = cross_val_score(fit_pipeline, train_x, train_y_transformed, cv=5, scoring="accuracy")

# ##
# from sklearn.model_selection import cross_val_predict
# train_pred = cross_val_predict(fit_pipeline, train_x, train_y_transformed, cv=3)

# ##
# from sklearn.metrics import confusion_matrix
# error_matrix = confusion_matrix(train_y_transformed, train_pred)

## test_set 예측
test_set = pd.read_csv(test_dir)

##
test_set = test_set.drop("index", axis=1)

##
test_set_transformed = torch.tensor(x_pipeline.transform(test_set), dtype=torch.float32).to(device)

net.eval()
test_pred = fn_pred(net(test_set_transformed).detach().cpu().numpy())

##
class PredTransform(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x[x == 0.0] = 2
        x = x.astype(np.int64)
        x = x[:, np.newaxis]

        return x

##
pred_transform = PredTransform()
test_pred_transformed = pred_transform.transform(test_pred)

##
test_pred_transformed = np.squeeze(test_pred_transformed, axis=-1)

##
submission=pd.read_csv(sub_dir, index_col=0)
submission["voted"] = test_pred_transformed
submission.to_csv(os.path.join(result_dir, "my_submission.csv"))

# ## 모델 세부 튜닝
# from sklearn.model_selection import GridSearchCV

# param_grid = [{'attribs__age_group__seventy_trans': [70.0, 75.5, 80.0],
#                'attribs__education__other_trans': [1.0, 1.5, 2.0, 2.5, 3.0],
#                'attribs__married__other_trans': [1.0, 1.5, 2.0, 2.5],
#                'attribs__familysize__top_clip': [30, 35, 40, 45, 50]}]

# grid_search = GridSearchCV(fit_pipeline, param_grid, cv=3, scoring="accuracy", verbose=1)
# grid_search.fit(train_x, train_y_transformed)

# ##
# print(grid_search.best_params_)
# # {'attribs__age_group__seventy_trans': 75.5, 'attribs__education__other_trans': 2.5,
# # 'attribs__familysize__top_clip': 35, 'attribs__married__other_trans': 1.5}

# ##
# cvres = grid_search.cv_results_

# for mean_scores, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(mean_scores, params)

# ##

