
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import get_util

class Features(object):
    def __init__(self, data, sample_windows, label_windows, interval_day, slip_times, is_training=True):
        self.data = data
        self.sample_windows = sample_windows
        self.label_windows = label_windows
        self.interval_day = interval_day
        self.slip_times = slip_times
        self.is_training = is_training
        print("开始滑窗的初始时间结点为%s!"%(self.interval_day))
        print("用于滑动取特征的窗长为%s!" % (self.sample_windows))
        print("用于滑动取标签的窗长为%s!"%(self.label_windows))
        print("滑动的次数为%s!"%(self.slip_times))
        print("是否为训练集%s!"%(self.is_training))
        print("======定义用于训练的标签名称+ID名称======")
        # 定义标签
        # s1标签为标签提取窗长中订单个数
        # s2标签为标签提取窗长中最早购买订单与特征提取窗长中最近购买订单之间的天数差距
        self.LabelColumns = ['label_30_101_BuyNums', 'label_30_101_IntervalDay']
        self.IDColumns = ['user_id']
        print("标签名称为：%s!" % (";".join(self.LabelColumns)))
        print("ID名称为：%s!" % (";".join(self.IDColumns)))
        print("======拼接原始信息======")
        # 拼接原始信息
        # 1)订单信息为主
        # order comment user sku
        # Index(['user_id', 'sku_id', 'o_id', 'o_date', 'o_area', 'o_sku_num',
        #        'year_order', 'month_order', 'day_order', 'comment_create_tm',
        #        'score_level', 'year_comment', 'month_comment', 'day_comment', 'age',
        #        'sex', 'user_lv_cd', 'price', 'cate', 'para_1', 'para_2', 'para_3'],
        #       dtype='object')
        self.Order_Comment_User_Sku = self.data['order']. \
            merge(self.data['comment'], on=['user_id', 'o_id'], how='left'). \
            merge(self.data['user'], on=['user_id'], how='left'). \
            merge(self.data['sku'], on=['sku_id'], how='left')
        self.Order_Comment_User_Sku = self.Order_Comment_User_Sku.rename(
            columns={'year_x': 'year_order', 'month_x': 'month_order', 'day_x': 'day_order',
                     'year_y': 'year_comment', 'month_y': 'month_comment', 'day_y': 'day_comment'})
        # 2)行为信息为主
        # action user sku
        # Index(['user_id', 'sku_id', 'a_date', 'a_num', 'a_type', 'year', 'month',
        #        'day', 'age', 'sex', 'user_lv_cd', 'price', 'cate', 'para_1', 'para_2',
        #        'para_3'],
        #       dtype='object')
        self.Action_User_Sku = self.data['action']. \
            merge(self.data['user'], on=['user_id'], how='left'). \
            merge(self.data['sku'], on=['sku_id'], how='left')
        print("完成原始信息拼接!")
        self.Sample_datas = pd.DataFrame()
        for i in range(slip_times):
            for j in self.sample_windows:
                sample_date, label_date = self.get_data_slip(i, j)
                # 判断是否是促销日：11.11/6.18
                # 只按照30天对获取特征部分进行滑窗
                if j == 30:
                    print("======获取标签信息======")
                    self.Sample_data = self.get_label(sample_date, label_date)
                    print(self.Sample_data.shape[0])
            self.Sample_datas = pd.concat([self.Sample_datas, self.Sample_data])
            # 加入特征：对用于打标签的特征窗中的行为时间
            print(self.Sample_datas.columns)
            print("该次整体滑窗后特征数为%s条" % (self.Sample_datas.shape[1]))
            print("该次整体滑窗后总样本数为%s条" % (self.Sample_datas.shape[0]))
            print("========================一次滑窗结束============================")


    def get_data_slip(self, slip_time, sample_window):
        # slip_time滑窗第几次，从0开始的
        # 上一次滑窗及之前的划窗移动天数
        sliped_days = timedelta(days=slip_time*(self.label_windows+1))
        # 用于提取特征的滑窗开始时间与结束时间
        sample_start_time = self.interval_day - sliped_days - timedelta(days=sample_window)
        sample_end_time = self.interval_day - sliped_days - timedelta(days=1)
        # 用于提取标签的滑窗开始时间与结束时间
        label_start_time = self.interval_day - sliped_days
        label_end_time = self.interval_day - sliped_days + timedelta(days=self.label_windows)
        print("用于提取特征的滑窗开始时间%s与结束时间%s"%(str(sample_start_time)[:10], str(sample_end_time)[:10]))
        print("用于提取标签的滑窗开始时间%s与结束时间%s"%(str(label_start_time)[:10], str(label_end_time)[:10]))
        # 滑窗取得的所有具体时间
        sample_date = [str(d)[:10] for d in pd.date_range(sample_start_time, sample_end_time)]
        label_date = [str(d)[:10] for d in pd.date_range(label_start_time, label_end_time)]
        return sample_date, label_date

    def get_label(self, sample_date, label_date):
        self.Sample_data = self.data['user']
        # 注意, s1标签与订单有关, s2标签与行为有关
        # 提取特征时间段中有行为但是提取标签时间段中没有订单则为0, 反之为有订单则为对应订单数
        # 行为在提取特征时间段中的部分
        Action_Sample = self.Action_User_Sku[\
                (self.Action_User_Sku['a_date'] >= sample_date[0])& \
                (self.Action_User_Sku['a_date'] <= sample_date[-1])]
        # 订单在提取标签时间段中的部分
        Order_Label = self.Order_Comment_User_Sku[ \
            (self.Order_Comment_User_Sku['o_date'] >= label_date[0]) & \
            (self.Order_Comment_User_Sku['o_date'] <= label_date[-1])]
        # 提取标签时间段的购买品类30和101的订单数据
        Order_Label_30_101 = Order_Label[(Order_Label['cate']==30) | (Order_Label['cate']==101)]
        # 提取特征时间段的购买品类30和101的订单数据
        # Order_Sample_30_101 = Order_Samples[(Order_Samples['cate']==30) | (Order_Samples['cate']==101)]
        Order_Sample_30_101 = Action_Sample
        if(self.is_training==True):
            # 1)统计标签时间段中每个用户购买品类30或者101的总订单数
            # groupby里面的字段都会变成索引, 先按照user_id分组, 然后将o_id种类数叠加, 最后结果只有user_id+o_id两列
            BuyNum_30_101_Pre = Order_Label_30_101.groupby(['user_id'])['o_id'].nunique(). \
                reset_index().rename(columns={'user_id': 'user_id', 'o_id': 'label_30_101_BuyNums'})
            # 2)统计标签时间段中每个用户购买品类30或者101的最早日期
            # 先按照user_id排序, 再按照o_date排序, 保留user_id中的第一条数据
            FirstTime_30_101_Pre = Order_Label_30_101.sort_values(['user_id', 'o_date']). \
                drop_duplicates('user_id', keep='first')[['user_id', 'o_date']]. \
                rename(columns={'user_id': 'user_id', 'o_date': 'label_30_101_FirstTime'})
            # 3)统计特征时间段中每个用户行为的最近日期
            LastTime_30_101_Feature = Order_Sample_30_101.sort_values(['user_id', 'a_date']). \
                drop_duplicates('user_id', keep='last')[['user_id', 'a_date']]. \
                rename(columns={'us.label_30_101_BuyNumser_id': 'user_id', 'a_date': 'Sample_30_101_LastTime'})
            self.Sample_data = self.Sample_data.merge(FirstTime_30_101_Pre, on=['user_id'], how='inner').\
                merge(LastTime_30_101_Feature, on=['user_id'], how='inner').\
                merge(BuyNum_30_101_Pre, on=['user_id'], how='left')
            # # 将购买数量改为是否有购买行为
            # self.Sample_data.label_30_101_BuyNums = self.Sample_data.label_30_101_BuyNums\
            #     .map(lambda x:1 if x>0 else 0)
            self.Sample_data['label_30_101_IntervalDay'] = (self.Sample_data['label_30_101_FirstTime']-
                self.Sample_data['Sample_30_101_LastTime']).dt.days
            print("该批次训练样本中共有%s条正例训练数据"%(self.Sample_data.shape[0]))
            # 4)添加负例, 其中label_30_101_BuyNums设置为0, label_30_101_IntervalDay设置为100
            # 将所有用户中没有参与标签提取月购买, 或者参与标签提取月购买但是特征提取月没有上次行为的用户提取出来
            user_negative = self.data['user'][~((self.data['user']['user_id']).isin(self.Sample_data['user_id']))]
            user_negative['label_30_101_BuyNums'] = 0
            user_negative['label_30_101_IntervalDay'] = 100
            print("该批次训练样本中共有%s条负例训练数据" % (user_negative.shape[0]))
            print("该批次训练样本中负例与正例之比为(%s:1)!"%
                  (user_negative.shape[0]/self.Sample_data.shape[0]))
            self.Sample_data = pd.concat([self.Sample_data, user_negative]).sort_values(['user_id'])
            # 取消掉Sample_30_101_LastTime, label_30_101_FirstTime这两个无关标签
            self.Sample_data = self.Sample_data.drop(['label_30_101_FirstTime'], axis=1)
        else:
            LastTime_30_101_Feature = Order_Sample_30_101.sort_values(['user_id', 'a_date']). \
                drop_duplicates('user_id', keep='last')[['user_id', 'a_date']]. \
                rename(columns={'user_id': 'user_id', 'a_date': 'Sample_30_101_LastTime'})
            self.Sample_data = self.Sample_data.merge(LastTime_30_101_Feature, on=['user_id'], how='left')
            self.Sample_data['label_30_101_BuyNums'] = -1
            self.Sample_data['label_30_101_IntervalDay'] = -1
        # Index(['user_id', 'sku_id', 'a_date', 'a_num', 'a_type', 'year', 'month',
        #        'day', 'age', 'sex', 'user_lv_cd', 'price', 'cate', 'para_1', 'para_2',
        #        'para_3'],
        #       dtype='object')
        print(self.Sample_data.columns)
        return self.Sample_data


