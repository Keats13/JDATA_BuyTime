import datetime
import random
from pyecharts import HeatMap
import numpy as np
import get_util
import get_cleaning as gc

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

print("======订单日历图绘制======")
# 统计每日订单量
# 删减掉对应用户
user_cleaning = gc.user_ctotal
user_order = user_order[~(user_order['user_id'].isin(user_cleaning['user_id']))]
day_order_df = user_order.groupby(['o_date'])['o_id'].nunique().reset_index().\
        rename(columns={'o_date':'o_date','o_id':'o_id_nunique'})
# user_action = user_action[~(user_action['user_id'].isin(user_cleaning['user_id']))]
# day_order_df = user_action.groupby(['a_date'])['a_num'].count().reset_index().\
#         rename(columns={'a_date':'a_date','a_num':'o_id_nunique'})
day_order = np.array(day_order_df).tolist()
day_order_sort = day_order_df.sort_values('o_id_nunique')
print(day_order_sort)
heatmap = HeatMap("日历热力图示例", "JDATA B榜每日行为总量", width=2000)
heatmap.add("", day_order, is_calendar_heatmap=True,
            visual_text_color='#000', visual_range_text=['', ''],
            visual_range=[200, 6000], calendar_cell_size=['auto', 40],
            is_visualmap=True, calendar_date_range= ["2016-9-1", "2017-8-31"],
            visual_orient="horizontal", visual_pos="center",
            is_piecewise=True, visual_split_number=20)
heatmap.render("JData_action_b.html")

# 286 2017-06-14          4808
# 322 2017-07-20          5052
# 341 2017-08-08          5278
# 278 2017-06-06          5471
# 273 2017-06-01          5601
# 102 2016-12-12          5926
# 222 2017-04-11          5996
# 292 2017-06-20          6061
# 288 2017-06-16          6299
# 285 2017-06-13          6519
# 281 2017-06-09          6530
# 284 2017-06-12          6943
# 291 2017-06-19          7059
# 289 2017-06-17          9384
# 71  2016-11-11         13194
# 290 2017-06-18         30437

# user_cleaning = gc.user_buy_1_or_2
# user_order = user_order[~(user_order['user_id'].isin(user_cleaning['user_id']))]
# user_order_sku = user_order.merge(sku_basic_info, on='sku_id', how='left')
# user_order = user_order_sku[(user_order_sku['month']==(6+1)) | (user_order_sku['month']==(4+1)) | (user_order_sku['month']==(3+1)) |
#     (user_order_sku['month'] == (2+ 1)) ]
# user_order_30_101 = user_order
# # user_order_4_30_101 = user_order_4[(user_order_4['cate']==30) | (user_order_4['cate']==101)]
# user_order_sorted = user_order_30_101.sort_values(by=['user_id','o_date'], ascending=True)
# user_drop = user_order_sorted.drop_duplicates('user_id')
# # 总购买用户71405个
# print("总购买用户%s个！"%(user_drop.shape[0]))

