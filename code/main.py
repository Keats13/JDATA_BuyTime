import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta, date
import get_features
import get_util
warnings.filterwarnings('ignore')

# 将s1标签转换为距离时间点的间隔天数

print("======原始数据信息导入======")
sku_basic_info_Path = "../data/jdata_sku_basic_info.csv"
user_basic_info_Path = "../data/jdata_user_basic_info.csv"
user_action_Path = "../data/jdata_user_action.csv"
user_order_Path = "../data/jdata_user_order.csv"
user_comment_score_Path = "../data/jdata_user_comment_score.csv"

sku_basic_info = get_util.load_csv(sku_basic_info_Path)
user_basic_info = get_util.load_csv(user_basic_info_Path)
user_action = get_util.load_csv(user_action_Path, 'a_date')
user_order = get_util.load_csv(user_order_Path, 'o_date')
user_comment_score = get_util.load_csv(user_comment_score_Path, 'comment_create_tm')

print("======数据预处理----更改缺失值为空======")
sku_basic_info = get_util.change_null(sku_basic_info, ['para_1', 'para_2', 'para_3'])
user_basic_info = get_util.change_null(user_basic_info, ['age'])
user_comment_score = get_util.change_null(user_comment_score, ['score_level'])


data = {'sku':sku_basic_info,
        'user':user_basic_info,
        'action':user_action,
        'order':user_order,
        'comment':user_comment_score}


print("======指定训练集、验证集、测试集划分节点与相应参数======")
# 划分节点, 向前的数据用于获取特征, 向后的数据用于获取标签
train_interval=date(2017, 7, 1)
valid_interval=date(2017, 8, 1)
test_interval=date(2017, 9, 1)
sample_windows = [30, 60, 90, 180] # 用于提取特征的滑窗长度
label_windows = 30 # 用于验证标签的滑窗长度
slip_times = [0, 2, 3, 4] # 滑动取值次数


print("======对数据集进行滑动取值======")
features = get_features.Features(data, sample_windows, label_windows, train_interval, slip_times, is_training=True)
features.Sample_datas.to_csv("train_data.csv", index=0)
pre_features = get_features.Features(data, sample_windows, label_windows, test_interval, [0], is_training=False)
pre_features.Sample_datas.to_csv("test_data.csv", index=0)
# valid_features = get_features.Features(data, sample_windows, label_windows, valid_interval, [0], is_training=False)
# valid_features.Sample_datas.to_csv("valid_data.csv",index=0)



