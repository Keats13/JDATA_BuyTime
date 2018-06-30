import pandas as pd

def load_csv(csv_Path, parse_dates=None):
    if parse_dates!=None:
        csv_data = pd.read_csv(csv_Path, parse_dates=[parse_dates])
        # 对含有时间数据的信息进行排序
        csv_data = csv_data.sort_values(['user_id', parse_dates])
        # 添加年、月、日特征
        csv_data['year'] = csv_data[parse_dates].dt.year
        csv_data['month'] = csv_data[parse_dates].dt.month
        csv_data['day'] = csv_data[parse_dates].dt.day
    else:
        csv_data = pd.read_csv(csv_Path)
    return csv_data


# 特征构建函数
# 计算count
def feature_count(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].count().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算nunique
def feature_nunique(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].nunique().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算median
def feature_median(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].median().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算mean
def feature_mean(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].mean().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算sum
def feature_sum(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].sum().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算max
def feature_max(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].max().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算min
def feature_min(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].min().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 计算std
def feature_std(features, message, f_col, f_name):
    feature = message.groupby(['user_id'])[f_col].std().reset_index().\
        rename(columns={'user_id':'user_id', f_col:f_name})
    features = features.merge(feature, on=['user_id'], how='left')
    return features

# 最近一次操作距离时间节点的时间差
def feature_action_timeGap(features, message, f_col, f_name, timePoint):
    feature = message.sort_values(['user_id', f_col]). \
        drop_duplicates('user_id', keep='last')[['user_id',f_col]]. \
        rename(columns={'user_id': 'user_id',f_col: f_name})
    features = features.merge(feature, on='user_id', how='left')
    features[f_name] = features[f_name].apply(lambda x:(timePoint-x).days)
    return features

# 最近一次操作信息
# 订单：'price', 'cate', 'para_1', 'para_2', 'para_3', 'o_area', 'o_sku_num'
def feature_last_order(features, message, f_col, f_cols, monthTag, chooseCate):
    feature = message.sort_values(['user_id', f_col]). \
        drop_duplicates('user_id', keep='last')[['user_id']+f_cols]
    feature.columns = feature.columns.map(lambda x:(str(monthTag)+x+chooseCate) if x!='user_id' else x)
    features = features.merge(feature, on='user_id', how='left')
    return features


# 将枚举信息转为one-hot形式
def feature_onehot(features, f_cols):
    features = pd.get_dummies(features, columns=f_cols)
    return features

# 将缺失信息的-1值转换为空值
# 列包括para_1,para_2,para_3,age
def change_null(message, f_cols):
    for f_col in f_cols:
        message[f_col] = message[f_col].apply(lambda x:None if x==-1 else x)
    return message

# 计算操作日期与时间节点之间的差值, 再进行其他计算
def feature_op_gap(message, f_col, f_name, timePoint):
    message['timePoint'] = timePoint
    message[f_name] = (message['timePoint'] - message[f_col]).dt.days
    feature = message[['user_id', f_name]]
    return feature

# 计算评论日期与订单日期的时间差值, 再进行其他计算
def feature_time_gap(message, f_col1, f_col2, f_name):
    message[f_name] = (message[f_col1] - message[f_col2]).dt.days
    feature = message[['user_id', f_name]]
    return feature

# 某类订单比例=某类订单数/总订单数
def feature_count_rate(features, message1, message2, f_col, f_name):
    feature_cate = message1.groupby(['user_id'])[f_col].count().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'one_cate_num'})
    feature_all = message2.groupby(['user_id'])[f_col].count().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'all_cates_num'})
    # user_id, one_cate_num, all_cates_num
    feature = feature_all.merge(feature_cate, on='user_id', how='left')
    feature[f_name] = feature['one_cate_num']/feature['all_cates_num']
    feature = feature[['user_id', f_name]]
    features = features.merge(feature, on='user_id', how='left')
    return features

def feature_nunique_rate(features, message1, message2, f_col, f_name):
    feature_cate = message1.groupby(['user_id'])[f_col].nunique().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'one_cate_num'})
    feature_all = message2.groupby(['user_id'])[f_col].nunique().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'all_cates_num'})
    # user_id, one_cate_num, all_cates_num
    feature = feature_all.merge(feature_cate, on='user_id', how='left')
    feature[f_name] = feature['one_cate_num']/feature['all_cates_num']
    feature = feature[['user_id', f_name]]
    features = features.merge(feature, on='user_id', how='left')
    return features


def feature_buy_second_rate(features, message, f_col, f_name):
    message_tmp = message[['user_id','o_date','o_sku_num']]
    o_sku_num_all = message_tmp.groupby(['user_id'])[f_col].sum().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'o_sku_num_all'})
    # 去除该品类第一次购买的订单信息
    message_first = message_tmp.sort_values(['user_id','o_date'], ascending=True).\
        drop_duplicates(['user_id'], keep='first')
    message_tmp = message_tmp.append(message_first)
    message_tmp = message_tmp.append(message_first)
    message_sencond = message_tmp.drop_duplicates(keep=False)
    o_sku_num_second = message_sencond.groupby(['user_id'])[f_col].sum().reset_index(). \
        rename(columns={'user_id': 'user_id', f_col: 'o_sku_num_second'})
    o_sku_num = o_sku_num_all.merge(o_sku_num_second, on='user_id', how='left')
    o_sku_num[f_name] = o_sku_num['o_sku_num_second']/o_sku_num['o_sku_num_all']
    feature = o_sku_num[['user_id',f_name]]
    features = features.merge(feature, on='user_id', how='left')
    return features

def feature_order_sale_rate(features, message, f_col, f_name, sale_days):
    # 促销日中的信息
    message_sale_days = message[message['o_date'].isin(sale_days)]
    # 用户在促销日的操作数量(订单/行为/浏览/收藏)
    message_sale_nums = message_sale_days.groupby(['user_id'])[f_col].nunique().reset_index().\
        rename(columns={'user_id':'user_id', f_col:'sale_days_nums'})
    # 用户总的操作数量
    message_nums = message.groupby(['user_id'])[f_col].nunique().reset_index().\
        rename(columns={'user_id':'user_id', f_col:'total_days_nums'})
    message_tmp = message_nums.merge(message_sale_nums, on='user_id', how='left')
    # 促销日比例
    message_tmp[f_name] = message_tmp['sale_days_nums']/message_tmp['total_days_nums']
    message_tmp = message_tmp[['user_id', f_name]]
    features = features.merge(message_tmp, on='user_id', how='left')
    return features

def feature_action_sale_rate(features, message, f_col, f_name, sale_days):
    # 促销日中的信息
    message_sale_days = message[message['a_date'].isin(sale_days)]
    # 用户在促销日的操作数量(订单/行为/浏览/收藏)
    message_sale_nums = message_sale_days.groupby(['user_id'])[f_col].sum().reset_index().\
        rename(columns={'user_id':'user_id', f_col:'sale_days_nums'})
    # 用户总的操作数量
    message_nums = message.groupby(['user_id'])[f_col].sum().reset_index().\
        rename(columns={'user_id':'user_id', f_col:'total_days_nums'})
    message_tmp = message_nums.merge(message_sale_nums, on='user_id', how='left')
    # 促销日比例
    message_tmp[f_name] = message_tmp['sale_days_nums']/message_tmp['total_days_nums']
    message_tmp = message_tmp[['user_id', f_name]]
    features = features.merge(message_tmp, on='user_id', how='left')
    return features


# 最近一次行为是否是促销日
def feature_last_action_sale(features, message, f_col, f_name, sale_days):
    feature = message.sort_values(['user_id', f_col]). \
        drop_duplicates('user_id', keep='last')[['user_id',f_col]]
    # 如果为促销日则标记为1, 否则标记为0
    feature[f_name] = feature[f_col].apply(lambda x:1 if x in sale_days else 0)
    feature = feature[['user_id',f_name]]
    features = features.merge(feature, on='user_id', how='left')
    return features