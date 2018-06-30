import numpy as np
import pandas as pd
import get_util
import warnings
warnings.filterwarnings('ignore')

sku_basic_info_Path = "../data/jdata_sku_basic_info.csv"
user_order_Path = "../data/jdata_user_order.csv"

sku_basic_info = get_util.load_csv(sku_basic_info_Path)
user_order = get_util.load_csv(user_order_Path, 'o_date')

# print("种子2018, 未删减18个用户, 576特征去掉了差评率,复购率,最大购买量到时间节点的时长, 用户购买为周末的比例")
# 验证
# 评分函数
def score(pred, real):
    # pred: user_id, pre_date | real: user_id, o_date
    # wi与oi的定义与官网相同
    pred['pred_day'] = pd.to_datetime(pred['pred_date']).dt.day
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    real['real_day'] = pd.to_datetime(real['o_date']).dt.day
    real['oi'] = 1

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare.fillna(0, inplace=True)  # 实际上没有购买的用户，correct_for_S1列的值为nan，将其赋为0
    S1 = np.sum(compare['oi'] * compare['wi']) / np.sum(compare['wi'])

    compare_for_S2 = compare[compare['oi'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['pred_day'] - compare_for_S2['real_day']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)

result = pd.read_csv('../result/valid.csv')
# real_result = data['order'][data['order']['month'] == 5][['user_id', 'o_date']].sort_values(by=['user_id', 'o_date']).drop_duplicates()
# real_result = real_result.drop(real_result[real_result[['user_id']].duplicated()].index, axis=0)
# 找到该月在品类30和101中最早购买的订单日期
print("使用757个特征评测，训练使用7，5，4，3月")
order = user_order.merge(sku_basic_info, on='sku_id', how='left')
data_real = order[order['month']==8]
data_real = data_real[(data_real['cate'] == 30) | (data_real['cate'] == 101)]
real_result = data_real.sort_values(['user_id', 'o_date']). \
    drop_duplicates('user_id', keep='first')[['user_id', 'o_date']]
score(result[:50000], real_result[:50000])
