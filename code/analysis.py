# 分析四月份下单用户
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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

# user_order_sku = user_order.merge(sku_basic_info, on='sku_id', how='left')
# for month in range(12):
#     user_order = user_order_sku[user_order_sku['month']==(month+1)]
#     user_order_30_101 = user_order
#     # user_order_4_30_101 = user_order_4[(user_order_4['cate']==30) | (user_order_4['cate']==101)]
#     user_order_sorted = user_order_30_101.sort_values(by=['user_id','o_date'], ascending=True)
#     user_drop = user_order_sorted.drop_duplicates('user_id')
#     print("%s月购买用户%s个！"%((month+1),user_drop.shape[0]))
#B榜
# 1月购买用户26128个！
# 2月购买用户29802个！
# 3月购买用户34452个！
# 4月购买用户37094个！
# 5月购买用户37597个！
# 6月购买用户62503个！
# 7月购买用户45824个！
# 8月购买用户50610个！
# 9月购买用户18677个！
# 10月购买用户22235个！
# 11月购买用户27090个！
# 12月购买用户24380个！

# A榜
# 1月购买用户35211个！
# 2月购买用户51568个！
# 3月购买用户56907个！
# 4月购买用户57658个！
# 5月购买用户17731个！
# 6月购买用户20582个！
# 7月购买用户18169个！
# 8月购买用户20794个！
# 9月购买用户23941个！
# 10月购买用户28896个！
# 11月购买用户35464个！
# 12月购买用户32977个！

user_cleaning = gc.user_ctotal
user_order = user_order[~(user_order['user_id'].isin(user_cleaning['user_id']))]
user_order_sku = user_order.merge(sku_basic_info, on='sku_id', how='left')
user_order_sku = user_order_sku[(user_order_sku['cate']==30) | (user_order_sku['cate']==101)]
# 大于某个时间点
time_start = datetime(year=2017,month=7,day=15)
time_end = datetime(year=2017,month=8,day=15)
user_order_time = user_order_sku[(user_order_sku['o_date']>=time_start) & (user_order_sku['o_date']<=time_end)]
user_order_time = user_order_time[['user_id','o_date','o_id']]
# 首次购买的用户
user_buy_first = user_order_time.sort_values(['user_id','o_date','o_id'],ascending=True)
user_buy_first = user_buy_first.drop_duplicates('user_id',keep='first')
user_order_time = user_order_time.append(user_buy_first)
user_order_time = user_order_time.append(user_buy_first)
user_buy_second = user_order_time.drop_duplicates(keep=False)
print("时间段内首次购买的用户有%s个！"%user_buy_first.shape[0])
user_buy_second = user_buy_second.sort_values(['user_id','o_date'],ascending=True).drop_duplicates('user_id',keep='first')
# a = user_buy_first.merge(user_buy_second, on='user_id', how='inner')
# a['gap'] = a['o_date_y']-a['o_date_x']
# aa = a[a['gap']==timedelta(0)]
# print(aa)
user_buy_first = user_buy_first[['user_id', 'o_date']].rename(columns={'user_id':'user_id', 'o_date':'o_date_first'})
user_buy_second = user_buy_second[['user_id', 'o_date']].rename(columns={'user_id':'user_id', 'o_date':'o_date_second'})
user_buy_tag = user_buy_first.merge(user_buy_second, on='user_id', how='inner')
user_buy_tag['time_tag'] = user_buy_tag['o_date_second']-user_buy_tag['o_date_first']
user_buy_tag = user_buy_tag[['user_id', 'time_tag']]
print("时间段内再次购买的用户有%s个！"%(user_buy_tag.shape[0]))
user_buy_tag_count = user_buy_tag.groupby('time_tag')['user_id'].count().reset_index().\
    rename(columns={'user_id':'user_count','time_tag':'time_tag'})
user_buy_tag_count['user_count'] = user_buy_tag_count['user_count'].apply(lambda x:x/float(user_basic_info.shape[0]))
user_buy_tag_count['time_tag'] = user_buy_tag_count['time_tag'].apply(lambda x:int(x.days))
user_buy_tag_count.plot(x='time_tag',y='user_count')
plt.show()
