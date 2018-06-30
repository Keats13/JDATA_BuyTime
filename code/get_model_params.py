import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import get_cleaning as gc
import warnings
warnings.filterwarnings('ignore')

binary = False
evaluate = False
train_data = pd.read_csv('train_data.csv')
if evaluate == True:
    test_data = pd.read_csv('valid_data.csv')
else:
    test_data = pd.read_csv('test_data.csv')

# user_cleaning = gc.user_cleaning
# # 对数据做筛减
# train_data = train_data[~train_data['user_id'].isin(user_cleaning)]
# test_data = test_data[~test_data['user_id'].isin(user_cleaning)]

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
features = [col for col in train_data.columns if
            (col not in IDColumns + LabelColumns + ['Sample_30_101_LastTime'])]
# cols = IDColumns + LabelColumns + features

print("======训练s1模型=======")

print("构建数据集")
label_1 = 'label_30_101_BuyNums'
train_features = train_data[features].values
train_label1 = train_data[label_1].values
# 将数据划分为训练集+验证集
train_x, valid_x, train_y, valid_y = train_test_split(train_features, train_label1,
                                                    test_size=0.2, random_state=0)
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_eval = lgb.Dataset(valid_x, label=valid_y, reference=lgb_train)

if evaluate==True:
    test_x = test_data[features].values
else:
    test_x = test_data[features].values

print("训练模型")
### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
	'boosting_type': 'gbdt',
	'objective': 'regression',
	'metric': 'auc',
	'num_leaves': 20,
	'learning_rate': 0.05,
	'feature_fraction': 0.9,
	'bagging_fraction': 0.8,
	'bagging_freq': 5,
	'verbose': 0,
	'seed': 2018,
	'min_child_weight': 1.5,
	'lambda_l2': 10,
	'scale_pos_weight': 20
    }

### 交叉验证(调参)
print('交叉验证')
max_merror = float(0)
best_params = {}

f1 = open('params1.txt','a+')
# 准确率
print("调参1：提高准确率")
for num_leaves in range(20,200,10):
    for max_depth in range(3,8,1):
        print("num_leaves为%s,max_depth为%s!"%(num_leaves,max_depth))
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            metrics=['auc'],
                            num_boost_round=10000,
                            early_stopping_rounds=30,
                            verbose_eval=True
                            )

        mean_merror = pd.Series(cv_results['l2-mean']).max()

        if mean_merror > max_merror:
            max_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth


print(best_params)
params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']
f1.write('num_leaves'+':'+str(params['num_leaves']))
f1.write('max_depth'+':'+str(params['num_leaves']))

# # # 过拟合
# # print("调参2：降低过拟合")
# # for max_bin in range(1,255,5):
# #     for min_data_in_leaf in range(10,200,5):
# #             params['max_bin'] = max_bin
# #             params['min_data_in_leaf'] = min_data_in_leaf
# #
# #             cv_results = lgb.cv(
# #                                 params,
# #                                 lgb_train,
# #                                 seed=2018,
# #                                 nfold=3,
# #                                 metrics=['auc'],
# #                                 early_stopping_rounds=15,
# #                                 verbose_eval=True
# #                                 )
# #
# #             mean_merror = pd.Series(cv_results['auc-mean']).min()
# #             boost_rounds = pd.Series(cv_results['auc-mean']).argmin()
# #
# #             if mean_merror < min_merror:
# #                 min_merror = mean_merror
# #                 best_params['max_bin']= max_bin
# #                 best_params['min_data_in_leaf'] = min_data_in_leaf
# #
# # params['min_data_in_leaf'] = best_params['min_data_in_leaf']
# # params['max_bin'] = best_params['max_bin']
#
# print("调参3：降低过拟合")
# for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
#     for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
#         for bagging_freq in range(0,20,5):
#             print("feature_fraction为%s，bagging_fraction为%s，bagging_freq为%s"
# 				  %(feature_fraction,bagging_fraction,bagging_freq))
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=2018,
#                                 nfold=3,
#                                 metrics=['auc'],
#                                 early_stopping_rounds=15,
#                                 verbose_eval=True
#                                 )
#
#             mean_merror = pd.Series(cv_results['auc-mean']).max()
#
#             if mean_merror > max_merror:
#                 max_merror = mean_merror
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq
#
# params['feature_fraction'] = best_params['feature_fraction']
# params['bagging_fraction'] = best_params['bagging_fraction']
# params['bagging_freq'] = best_params['bagging_freq']
# print(params)
# f1.write('feature_fraction'+':'+str(params['feature_fraction']))
# f1.write('bagging_fraction'+':'+str(params['bagging_fraction']))
# f1.write('bagging_freq'+':'+str(params['bagging_freq']))
# #
#
#
# print("调参4：降低过拟合")
# for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#         for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#             print("lambda_l1为%s,lambda_l2为%s,min_split_gain为%s"%(lambda_l1,lambda_l2,min_split_gain))
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=2018,
#                                 nfold=3,
#                                 metrics=['auc'],
#                                 early_stopping_rounds=15,
#                                 verbose_eval=True
#                                 )
#
#             mean_merror = pd.Series(cv_results['auc-mean']).min()
#             boost_rounds = pd.Series(cv_results['auc-mean']).argmin()
#
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['lambda_l1'] = lambda_l1
#                 best_params['lambda_l2'] = lambda_l2
#                 best_params['min_split_gain'] = min_split_gain
#
# params['lambda_l1'] = best_params['lambda_l1']
# params['lambda_l2'] = best_params['lambda_l2']
# params['min_split_gain'] = best_params['min_split_gain']
# print(params)
# f1.write('lambda_l1'+':'+str(params['lambda_l1']))
# f1.write('lambda_l2'+':'+str(params['lambda_l2']))
# f1.write('min_split_gain'+':'+str(params['min_split_gain']))
#
# f1.close()
# print("==============================================================")
#
# print("======训练s2模型=======")
# print("构建数据集")
# label_2 = 'label_30_101_IntervalDay'
# # 只选择购买了的训练
# train_features = train_data[train_data[label_2]<100][features].values
# train_label2 = train_data[train_data[label_2]<100][label_2].values
#
# # 将数据划分为训练集+验证集
# train_x, valid_x, train_y, valid_y = train_test_split(train_features, train_label2,
#                                                     test_size=0.2, random_state=0)
# train_set = lgb.Dataset(train_x, label=train_y)
# valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set)
#
# if evaluate==True:
#     test_x = test_data[features].values
# else:
#     test_x = test_data[features].values
#
# print("训练模型")
# ### 设置初始参数--不含交叉验证参数
# print('设置参数')
# params = {
#           'boosting_type': 'gbdt',
#           'objective': 'regression',
#           'metric': 'l2',
#           }
#
# ### 交叉验证(调参)
# print('交叉验证')
# min_merror = float('Inf')
# best_params = {}
# f2 = open('params2.txt','a+')
#
# # 准确率
# print("调参1：提高准确率")
# for num_leaves in range(20,100,5):
#     for max_depth in range(3,8,1):
#         print("num_leaves为%s,max_depth为%s!"%(num_leaves,max_depth))
#         params['num_leaves'] = num_leaves
#         params['max_depth'] = max_depth
#
#         cv_results = lgb.cv(
#                             params,
#                             lgb_train,
#                             seed=2018,
#                             nfold=3,
#                             metrics=['l2'],
#                             early_stopping_rounds=15,
#                             verbose_eval=True
#                             )
#
#         mean_merror = pd.Series(cv_results['l2-mean']).min()
#         boost_rounds = pd.Series(cv_results['l2-mean']).argmin()
#
#         if mean_merror < min_merror:
#             min_merror = mean_merror
#             best_params['num_leaves'] = num_leaves
#             best_params['max_depth'] = max_depth
#
# params['num_leaves'] = best_params['num_leaves']
# params['max_depth'] = best_params['max_depth']
# print(params)
# f2.write('num_leaves'+':'+str(params['num_leaves']))
# f2.write('max_depth'+':'+str(params['num_leaves']))

# # # 过拟合
# # print("调参2：降低过拟合")
# # for max_bin in range(1,255,5):
# #     for min_data_in_leaf in range(10,200,5):
# #             params['max_bin'] = max_bin
# #             params['min_data_in_leaf'] = min_data_in_leaf
# #
# #             cv_results = lgb.cv(
# #                                 params,
# #                                 lgb_train,
# #                                 seed=2018,
# #                                 nfold=3,
# #                                 metrics=['l2'],
# #                                 early_stopping_rounds=15,
# #                                 verbose_eval=True
# #                                 )
# #
# #             mean_merror = pd.Series(cv_results['l2-mean']).min()
# #             boost_rounds = pd.Series(cv_results['l2-mean']).argmin()
# #
# #             if mean_merror < min_merror:
# #                 min_merror = mean_merror
# #                 best_params['max_bin']= max_bin
# #                 best_params['min_data_in_leaf'] = min_data_in_leaf
# #
# # params['min_data_in_leaf'] = best_params['min_data_in_leaf']
# # params['max_bin'] = best_params['max_bin']
#
# print("调参3：降低过拟合")
# for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
#     for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
#         for bagging_freq in range(0,20,5):
#             print("feature_fraction为%s，bagging_fraction为%s，bagging_freq为%s"
# 				  %(feature_fraction,bagging_fraction,bagging_freq))
#             params['feature_fraction'] = feature_fraction
#             params['bagging_fraction'] = bagging_fraction
#             params['bagging_freq'] = bagging_freq
#
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=2018,
#                                 nfold=3,
#                                 metrics=['l2'],
#                                 early_stopping_rounds=15,
#                                 verbose_eval=True
#                                 )
#
#             mean_merror = pd.Series(cv_results['l2-mean']).min()
#             boost_rounds = pd.Series(cv_results['l2-mean']).argmin()
#
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['feature_fraction'] = feature_fraction
#                 best_params['bagging_fraction'] = bagging_fraction
#                 best_params['bagging_freq'] = bagging_freq
#
# params['feature_fraction'] = best_params['feature_fraction']
# params['bagging_fraction'] = best_params['bagging_fraction']
# params['bagging_freq'] = best_params['bagging_freq']
# print(params)
# f2.write('feature_fraction'+':'+str(params['feature_fraction']))
# f2.write('bagging_fraction'+':'+str(params['bagging_fraction']))
# f2.write('bagging_freq'+':'+str(params['bagging_freq']))
#
# print("调参4：降低过拟合")
# for lambda_l1 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     for lambda_l2 in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#         for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#             print("lambda_l1为%s,lambda_l2为%s,min_split_gain为%s"%(lambda_l1,lambda_l2,min_split_gain))
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#             cv_results = lgb.cv(
#                                 params,
#                                 lgb_train,
#                                 seed=2018,
#                                 nfold=3,
#                                 metrics=['l2'],
#                                 early_stopping_rounds=15,
#                                 verbose_eval=True
#                                 )
#
#             mean_merror = pd.Series(cv_results['l2-mean']).min()
#             boost_rounds = pd.Series(cv_results['l2-mean']).argmin()
#
#             if mean_merror < min_merror:
#                 min_merror = mean_merror
#                 best_params['lambda_l1'] = lambda_l1
#                 best_params['lambda_l2'] = lambda_l2
#                 best_params['min_split_gain'] = min_split_gain
#
# params['lambda_l1'] = best_params['lambda_l1']
# params['lambda_l2'] = best_params['lambda_l2']
# params['min_split_gain'] = best_params['min_split_gain']
# print(params)
# f2.write('lambda_l1'+':'+str(params['lambda_l1']))
# f2.write('lambda_l2'+':'+str(params['lambda_l2']))
# f2.write('min_split_gain'+':'+str(params['min_split_gain']))
#
# f2.close()
