import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from SBBTree_ONLINE import SBBTree
import warnings
warnings.filterwarnings('ignore')
import get_cleaning as gc
# pd.set_option('display.max_rows',500)

evaluate = False
binary = False

train_data = pd.read_csv('./train_data.csv')
valid_data = pd.read_csv('./valid_data.csv')
test_data = pd.read_csv('./test_data.csv')

user_cleaning = gc.user_ctotal
# 对数据做筛减
train_data = train_data[~train_data['user_id'].isin(user_cleaning['user_id'])]
test_data = test_data[~test_data['user_id'].isin(user_cleaning['user_id'])]

# s1标签为标签提取窗长中订单个数
# s2标签为标签提取窗长中最早购买订单与特征提取窗长中最近购买订单之间的天数差距
LabelColumns = ['label_30_101_BuyNums', 'label_30_101_IntervalDay']
IDColumns = ['user_id']
print("标签名称为：%s!" % (";".join(LabelColumns)))
print("ID名称为：%s!" % (";".join(IDColumns)))
TrainColumns = [col for col in train_data.columns if
                (col not in IDColumns + LabelColumns + ['Sample_30_101_LastTime'])]

train_features = TrainColumns
cols = IDColumns + LabelColumns + train_features

if binary==True:
    # 改成二分类模型
    # 将有购买订单数的部分改为正例1
    train_data.ix[train_data["label_30_101_BuyNums"] > 0, "label_30_101_BuyNums"] = 1
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 2016,
        'min_child_weight': 1.5,
        'lambda_l2': 10,
        'scale_pos_weight': 20
    }
else:
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'auc',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'seed': 2016,
        'min_child_weight': 1.5,
        'lambda_l2': 10,
        'scale_pos_weight': 20
    }
###############################################################
model = SBBTree(params=params, stacking_num=5, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000,
                early_stopping_rounds=200, train_features=train_features)

# train 下个月购买次数预测 回归模型
train_label_BuyNum = 'label_30_101_BuyNums'

train_features = train_features

X = train_data[train_features].values
y = train_data[train_label_BuyNum].values

X_pred = test_data[train_features].values
X_valid = valid_data[train_features].values

model.fit(X, y)
test_data[train_label_BuyNum] = model.predict(X_pred)
valid_data[train_label_BuyNum] = model.predict(X_valid)

print(test_data[train_label_BuyNum])

###############################################################
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 2016,
    'min_child_weight': 1.5,
    'lambda_l2': 10,
    'scale_pos_weight': 20
}
model = SBBTree(params=params, stacking_num=5, bagging_num=3,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200, train_features=train_features)

# train 当月首次购买时间预测 回归模型
train_label_FirstTime = 'label_30_101_IntervalDay'
# 只选择购买了的训练
train_features = train_features


X = train_data[train_data[train_label_FirstTime]<100][train_features].values
y = train_data[train_data[train_label_FirstTime]<100][train_label_FirstTime].values

X_pred = test_data[train_features].values
X_valid = valid_data[train_features].values

model.fit(X,y)
test_data[train_label_FirstTime] = model.predict(X_pred)
valid_data[train_label_FirstTime] = model.predict(X_valid)
print(test_data[train_label_FirstTime])
####################################################################
# submit
columns = ['user_id'] + [train_label_BuyNum] + [train_label_FirstTime]
out_submit = test_data[columns].sort_values([train_label_BuyNum],ascending=False)
out_submit_valid = valid_data[columns].sort_values([train_label_BuyNum],ascending=False)
out_submit.to_csv('out.csv', index=0)

out_submit = pd.read_csv('out.csv')
train_label_FirstTime = 'label_30_101_IntervalDay'
out_submit['pred_date']=out_submit[train_label_FirstTime].map(lambda day: datetime(2017, 9, 1)+timedelta(days=int(day+0.49-1)))
print(out_submit.shape[0])
out_submit = out_submit[['user_id']+['pred_date']]
out_submit.head(50000).to_csv('../result/predict.csv',index=False,header=True)

out_submit_valid['pred_date']=out_submit_valid[train_label_FirstTime].map(lambda day: datetime(2017, 8, 1)+timedelta(days=int(day+0.49-1)))
out_submit_valid = out_submit_valid[['user_id']+['pred_date']]
out_submit_valid.head(50000).to_csv('../result/valid.csv',index=False,header=True)
