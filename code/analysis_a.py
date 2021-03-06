import datetime
import random
from pyecharts import HeatMap
import numpy as np
import pandas as pd
import get_util
import get_cleaning_a as gc

print("======原始数据信息导入======")
sku_basic_info_Path = "../data_a/jdata_sku_basic_info.csv"
user_basic_info_Path = "../data_a/jdata_user_basic_info.csv"
user_action_Path = "../data_a/jdata_user_action.csv"
user_order_Path = "../data_a/jdata_user_order.csv"
user_comment_score_Path = "../data_a/jdata_user_comment_score.csv"

sku_basic_info = get_util.load_csv(sku_basic_info_Path)
user_basic_info = get_util.load_csv(user_basic_info_Path)
user_action = get_util.load_csv(user_action_Path, 'a_date')
user_order = get_util.load_csv(user_order_Path, 'o_date')
user_comment_score = get_util.load_csv(user_comment_score_Path, 'comment_create_tm')

print("======订单日历图绘制======")
# 统计每日订单量
# 删减掉对应用户
user_cleaning = gc.user_buy_1_or_2
# user_order = user_order[~(user_order['user_id'].isin(user_cleaning['user_id']))]
# day_order_df = user_order.groupby(['o_date'])['o_id'].nunique().reset_index().\
#         rename(columns={'o_date':'o_date','o_id':'o_id_nunique'})
user_action = user_action[~(user_action['user_id'].isin(user_cleaning['user_id']))]
day_order_df = user_action.groupby(['a_date'])['a_num'].sum().reset_index().\
        rename(columns={'a_date':'a_date','a_num':'o_id_nunique'})
day_order = np.array(day_order_df).tolist()
day_order_sort = day_order_df.sort_values('o_id_nunique')
print(day_order_sort)
heatmap = HeatMap("日历热力图示例", "JDATA A榜每日订单总量", width=2000)
heatmap.add("", day_order, is_calendar_heatmap=True,
            visual_text_color='#000', visual_range_text=['', ''],
            visual_range=[11000, 200000], calendar_cell_size=['auto', 40],
            is_visualmap=True, calendar_date_range= ["2016-5-1", "2017-4-30"],
            visual_orient="horizontal", visual_pos="center",
            is_piecewise=True, visual_split_number=20)
heatmap.render("JData_action_a.html")

# 339 2017-04-05          3346
# 343 2017-04-09          3369
# 304 2017-03-01          3374
# 348 2017-04-14          3484
# 340 2017-04-06          3654
# 347 2017-04-13          3657
# 193 2016-11-10          3773
# 353 2017-04-19          3964
# 346 2017-04-12          4060
# 184 2016-11-01          4089
# 48  2016-06-18          5402
# 225 2016-12-12          6095
# 345 2017-04-11          6743
# 194 2016-11-11         12556

# user_order_sku = user_order.merge(sku_basic_info, on='sku_id', how='left')
# for month in range(12):
#     user_order = user_order_sku[user_order_sku['month']==(month+1)]
#     user_order_30_101 = user_order
#     # user_order_4_30_101 = user_order_4[(user_order_4['cate']==30) | (user_order_4['cate']==101)]
#     user_order_sorted = user_order_30_101.sort_values(by=['user_id','o_date'], ascending=True)
#     user_drop = user_order_sorted.drop_duplicates('user_id')
#     print("%s月购买用户%s个！"%((month+1),user_drop.shape[0]))

# user_cleaning = gc.user_buy_1_or_2
# user_order = user_order[~(user_order['user_id'].isin(user_cleaning['user_id']))]
# user_order_sku = user_order.merge(sku_basic_info, on='sku_id', how='left')
# user_order = user_order_sku[(user_order_sku['month']==(0+1)) | (user_order_sku['month']==(1+1)) | (user_order_sku['month']==(2+1)) |
#     (user_order_sku['month'] == (10 + 1))| (user_order_sku['month']==(11+1))]
# user_order_30_101 = user_order
# # user_order_4_30_101 = user_order_4[(user_order_4['cate']==30) | (user_order_4['cate']==101)]
# user_order_sorted = user_order_30_101.sort_values(by=['user_id','o_date'], ascending=True)
# user_drop = user_order_sorted.drop_duplicates('user_id')
# print("总购买用户%s个！"%(user_drop.shape[0]))# 82811