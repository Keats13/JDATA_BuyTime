import numpy as np
import pandas as pd
import warnings
import get_util
from datetime import datetime
warnings.filterwarnings('ignore')

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

user_order_cate = user_order.merge(sku_basic_info, on='sku_id', how='left')
print("======去掉只有购买记录,没有历史记录的用户======")
# 购买过的用户
user_buy = user_order['user_id'].drop_duplicates()
# 有行为的用户
user_action = user_action['user_id'].drop_duplicates()
# 有5471个用户
user_buy_noaction = list(set(user_buy).difference(set(user_action)))# 购买过但是没有行为的
# 统计其购买数
user_order_buy = user_order[user_order['user_id'].isin(user_buy_noaction)]
user_order_buy_num = user_order_buy.groupby(['user_id'])['o_id'].nunique()\
    .reset_index().sort_values(by=['o_id'], ascending=False)
user_buy_1 = user_order_buy_num[user_order_buy_num['o_id']==1]
user_buy_2 = user_order_buy_num[user_order_buy_num['o_id']==2]
user_buy_1_or_2 = pd.concat([user_buy_1, user_buy_2], axis=0)[['user_id']]
# 购买过一次的1837个，购买过两次的505个！
# 购买过一两次但是没有行为的用户总共2342个！
print("购买过一次的%s个，购买过两次的%s个！"%(user_buy_1.shape[0], user_buy_2.shape[0]))
print("购买过一两次但是没有行为的用户总共%s个！"%(user_buy_1_or_2.shape[0]))

# print("======去掉在用户表中, 但前三个月订单数据中并不存在的用户======")
# # 前三个月的订单数据
# user_order_cate = user_order_cate[(user_order_cate['cate']==30) | (user_order_cate['cate']==101)]
# order_three = user_order_cate[user_order_cate['month'].isin([2,3,4])]
# user_order_three = order_three['user_id'].drop_duplicates()
# user_all = user_basic_info['user_id'].drop_duplicates()
# user_no_inthree = list(set(user_all).difference(set(user_order_three)))
# print(user_no_inthree)
#
# # print("======去掉浏览量很大但是购买量很少的用户======")
# # # 用户浏览量
# # user_look_sum = user_action[user_action['a_type']==1].groupby(['user_id'])['a_num']\
# #     .sum().reset_index()
# # print(user_look_sum.shape[0])
# # # 用户购买量
# # user_order_sum = user_order.groupby(['user_id'])['o_id'].nunique().reset_index()
# # print(user_order_sum.shape[0])
# # user_look_order = user_order_sum.merge(user_look_sum, on=['user_id'],how='left')
# # user_look_order = user_look_order.sort_values(by='a_num', ascending=False)
# # # print(user_look_order[user_look_order['o_id'].isin([1,2,3])])
# # user_look_order['rate'] = user_look_order['a_num']/user_look_order['o_id']
# # user_inert = user_look_order[(user_look_order['rate']>=400) & (user_look_order['o_id']<=2)]['user_id'].drop_duplicates()
#
#
# # print("======将需要清洗掉的用户拼接起来======")
# # user_buy_1_or_2_list = np.array(user_buy_1_or_2['user_id']).tolist()
# # user_cleaning = user_buy_1_or_2_list + user_no_inthree
# # print("需要去掉的用户一共有%s个！"%(len(user_cleaning)))
# # print("其中只有购买记录,没有历史记录的用户有%s个！"%(len(user_buy_1_or_2_list)))
# # print("其中在用户表中, 但前三个月订单数据中并不存在的用户有%s个！"%(len(user_no_inthree)))


print("======只在618期间(6.12-6.20)进行购买的顾客=======")
user_order_618 = user_order[(user_order['o_date'].isin(pd.date_range(datetime(2017,6,1),datetime(2017,6,20)))) | \
                              (user_order['o_date'].isin([datetime(2016,11,11)])) ]
user_order_618 = user_order_618[['user_id']].drop_duplicates()# 13683
user_order_no618 = user_order[~((user_order['o_date'].isin(pd.date_range(datetime(2017,6,1),datetime(2017,6,20)))) |
                              (user_order['o_date'].isin([datetime(2016,11,11)])))]
user_order_no618 = user_order_no618[['user_id']].drop_duplicates()# 97420
# 在618买了，但没有在其他时候买过的
user_618noin_others = user_order_618[~(user_order_618['user_id'].isin(user_order_no618['user_id']))]# 2026
print("只在618进行了购物,在其他时候没有进行购物的顾客有%s个！"%(len(user_618noin_others)))

# print("======只在11月11日进行购买的顾客=======")
# user_order_1111 = user_order[user_order['o_date']==datetime(year=2016, month=11, day=11)]
# user_order_1111 = user_order_1111[['user_id']].drop_duplicates()
# print(user_order_1111)
# user_order_no1111 = user_order[user_order['o_date']!=datetime(year=2016, month=11, day=11)]
# user_order_no1111 = user_order_no1111[['user_id']].drop_duplicates()
# print(user_order_no1111)
# # 在1111买了，但没有在其他时候买过的
# user_1111noin_others = user_order_1111[~(user_order_1111['user_id'].isin(user_order_no1111['user_id']))]
# print("只在1111进行了购物,在其他时候没有进行购物的顾客有%s个！"%(len(user_1111noin_others)))

print("======汇总需要删掉的用户数======")
user_ctotal = user_buy_1_or_2.merge(user_618noin_others, on='user_id', how='outer')
print("总共需要删掉的用户有%s个"%(user_ctotal.shape[0]))

print("======将618当日购买的用户下采样======")

# ======去掉只有购买记录,没有历史记录的用户======
# 购买过一次的1837个，购买过两次的505个！
# 购买过一两次但是没有行为的用户总共2342个！
# ======只在618期间(6.12-6.20)进行购买的顾客=======
# 只在618进行了购物,在其他时候没有进行购物的顾客有5459个！
# ======汇总需要删掉的用户数======
# 总共需要删掉的用户有7481个