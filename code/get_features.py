
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import get_util
# pd.set_option('display.max_columns',100)

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
        for i in slip_times:
            for j in self.sample_windows:
                sample_date, label_date = self.get_data_slip(i, j)
                # 判断是否是促销日：11.11/6.18
                # 只按照30天对获取特征部分进行滑窗
                if j==30:
                    print("======获取标签信息======")
                    self.get_label(sample_date, label_date)
                    print(self.Sample_data.shape[0])
                    # TagExamine = 'E_'
                    # self.FuncFeatureExamine(TagExamine)
                print("======获取订单特征信息======")
                monthTagOrder = 'OC_'+str(j)+'slip_'
                self.FuncOrderFeatures(monthTagOrder, sample_date, label_date, j)
                print(self.Sample_data.shape[0])
                print("======获取评论特征信息======")
                monthTagComment = 'CC_'+str(j)+'slip_'
                self.FuncCommentFeatures(monthTagComment, sample_date, label_date, j)
                print(self.Sample_data.shape[0])
                print("======获取行为特征信息======")
                monthTagAction = 'AC_'+str(j)+'slip_'
                self.FuncActionFeatures(monthTagAction, sample_date, label_date, j)
                print(self.Sample_data.shape[0])
                print("该次特征滑窗后特征数为%s条" % (self.Sample_data.shape[1]))
                print("该次特征滑窗后总样本数为%s条" % (self.Sample_data.shape[0]))
                print("================================================")
            self.Sample_datas = pd.concat([self.Sample_datas, self.Sample_data])
            # 加入特征：对用于打标签的特征窗中的行为时间
            print("该次整体滑窗后特征数为%s条" % (self.Sample_datas.shape[1]))
            print("该次整体滑窗后总样本数为%s条" % (self.Sample_datas.shape[0]))
            print("========================一次滑窗结束============================")
        print("======将类别标签转换为onehot形式======")
        self.Sample_datas = self.FuncOnehotFeatures(self.Sample_datas)
        # print("======统计空值======")
        # self.Sample_datas = self.FuncNaNcount(self.Sample_datas)
        print("总特征数为%s条" % (self.Sample_datas.shape[1]))


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
        # 提取特征时间段中有行为但是提取标签时间段中没有订单则为0, 反之为有订单则为对应订单数
        # 订单在提取标签时间段中的部分
        Order_Label = self.Order_Comment_User_Sku[ \
            (self.Order_Comment_User_Sku['o_date'] >= label_date[0]) & \
            (self.Order_Comment_User_Sku['o_date'] <= label_date[-1])]
        # 提取标签时间段的购买品类30和101的订单数据
        Order_Label_30_101 = Order_Label[(Order_Label['cate']==30) | (Order_Label['cate']==101)]
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
            # 3）计算最早购买日期距离时间节点的天数
            FirstTime_30_101_Pre['label_30_101_IntervalDay'] = FirstTime_30_101_Pre['label_30_101_FirstTime'].\
                apply(lambda x:(x-datetime.strptime(label_date[0], "%Y-%m-%d")).days)
            self.Sample_data = self.Sample_data.merge(FirstTime_30_101_Pre, on=['user_id'], how='left').\
                merge(BuyNum_30_101_Pre, on=['user_id'], how='left')
            self.Sample_data['label_30_101_BuyNums'] = self.Sample_data['label_30_101_BuyNums'].fillna(0)
            self.Sample_data['label_30_101_IntervalDay'] = self.Sample_data['label_30_101_IntervalDay'].fillna(100)
            # 取消掉Sample_30_101_LastTime, label_30_101_FirstTime这两个无关标签
            self.Sample_data = self.Sample_data.drop(['label_30_101_FirstTime'], axis=1)
        else:
            # LastTime_30_101_Feature = Order_Sample_30_101.sort_values(['user_id', 'a_date']). \
            #     drop_duplicates('user_id', keep='last')[['user_id', 'a_date']]. \
            #     rename(columns={'user_id': 'user_id', 'a_date': 'Sample_30_101_LastTime'})
            # self.Sample_data = self.Sample_data.merge(LastTime_30_101_Feature, on=['user_id'], how='left')
            self.Sample_data['label_30_101_BuyNums'] = -1
            self.Sample_data['label_30_101_IntervalDay'] = -1
        # Index(['user_id', 'sku_id', 'a_date', 'a_num', 'a_type', 'year', 'month',
        #        'day', 'age', 'sex', 'user_lv_cd', 'price', 'cate', 'para_1', 'para_2',
        #        'para_3'],
        #       dtype='object')
        # print(self.Sample_data.columns)

    def FuncOrderFeatures(self, monthTag, sample_date, label_date, j):
        Order_Samples = self.Order_Comment_User_Sku[ \
            (self.Order_Comment_User_Sku['o_date'] >= sample_date[0]) & \
            (self.Order_Comment_User_Sku['o_date'] <= sample_date[-1])]
        # 提取特征时间段的购买品类30和101的订单数据
        Order_Sample_30_101 = Order_Samples[(Order_Samples['cate'] == 30) | (Order_Samples['cate'] == 101)]
        Order_Sample_30 = Order_Samples[Order_Samples['cate'] == 30]
        Order_Sample_101 = Order_Samples[Order_Samples['cate'] == 101]
        Order_Sample_others = Order_Samples[(Order_Samples['cate'] != 30) & (Order_Samples['cate'] != 101)]
        ##################################################################
        #  用户购买订单种类数(所有数据/30_101数据/30数据/101数据/其他类别)
        ##################################################################
        # 'OC_30slip_o_id_all_nunique','OC_30slip_o_id_30_101_nunique','OC_30slip_o_id_30_nunique','OC_30slip_o_id_101_nunique','OC_30slip_o_id_other_nunique',
        # 'OC_30slip_o_id_all_count','OC_30slip_o_id_30_101_count','OC_30slip_o_id_30_count','OC_30slip_o_id_101_count','OC_30slip_o_id_other_count'
        op_type = 'nunique'
        op_column = 'o_id'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                               str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        ##################################################################
        #     用户购买订单总数(所有数据/30_101数据/30数据/101数据/其他类别)
        ##################################################################
        # ['OC_30slip_o_id_all_count','OC_30slip_o_id_30_101_count']
        op_type = 'count'
        op_column = 'o_id'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        ##################################################################
        #       用户购买订单比例(30_101数据/30数据/101数据/其他类别)
        ##################################################################
        op_column = 'o_id'
        if j == 180:
            self.Sample_data = eval('get_util.feature_%s' % 'count_rate')\
                (self.Sample_data, Order_Sample_30_101, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_count_30_101'))
            self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
                (self.Sample_data, Order_Sample_30, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_count_30'))
            self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
                (self.Sample_data, Order_Sample_101, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_count_101'))
            self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
                (self.Sample_data, Order_Sample_others, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_count_other'))

            self.Sample_data = eval('get_util.feature_%s' % 'nunique_rate') \
                (self.Sample_data, Order_Sample_30_101, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_nunique_30_101'))
            self.Sample_data = eval('get_util.feature_%s' % 'nunique_rate') \
                (self.Sample_data, Order_Sample_30, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_nunique_30'))
            self.Sample_data = eval('get_util.feature_%s' % 'nunique_rate') \
                (self.Sample_data, Order_Sample_101, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_nunique_101'))
            self.Sample_data = eval('get_util.feature_%s' % 'nunique_rate') \
                (self.Sample_data, Order_Sample_others, Order_Samples, op_column,
                 str(monthTag + op_column + '_rate_nunique_other'))
        ##################################################################
        #              用户购买地区类别数(所有数据/30_101数据)
        ##################################################################
        # 'OC_30slip_o_area_all_nunique','OC_30slip_o_area_30_101_nunique'
        op_type = 'nunique'
        op_column = 'o_area'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        ##################################################################
        #  用户购买商品数信息(所有数据/30_101数据/30数据/101数据/其他类别)
        ##################################################################
        # 'OC_30slip_o_sku_num_all_sum', 'OC_30slip_o_sku_num_30_101_sum','OC_30slip_o_sku_num_30_sum','OC_30slip_o_sku_num_101_sum','OC_30slip_o_sku_num_other_sum'
        # 'OC_30slip_o_sku_num_all_mean', 'OC_30slip_o_sku_num_30_101_mean', 'OC_30slip_o_sku_num_30_mean', 'OC_30slip_o_sku_num_other_mean'
        op_type = 'sum'
        op_column = 'o_sku_num'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        ##################################################################
        #  用户购买天数(所有数据/30_101数据/30数据/101数据/其他类别)
        ##################################################################
        # 'OC_30slip_o_date_all_nunique', 'OC_30slip_o_date_30_101_nunique', 'OC_30slip_o_date_30_nunique', 'OC_30slip_o_date_101_nunique', 'OC_30slip_o_date_other_nunique'
        op_type = 'nunique'
        op_column = 'o_date'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        ##################################################################
        #    订单时间与时间节点之间的特征获取(所有数据/30_101数据)
        ##################################################################
        if j != 30:
            op_column = 'o_date'
            timePoint = datetime.strptime(label_date[0], '%Y-%m-%d')
            # 最小时间差
            op_type = 'min'
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 平均时间差
            op_type = 'mean'
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 方差
            op_type = 'std'
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Order_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
            # # 最大值
            # op_type = 'max'
            # df = eval('get_util.feature_%s' % 'op_gap')(Order_Sample_30_101, op_column,
            #                                             str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
            #                                             timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(
            #                                                              monthTag + 'mess_30_101_' + op_column + '_' + op_type),
            #                                                          str(
            #                                                              monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            # df = eval('get_util.feature_%s' % 'op_gap')(Order_Samples, op_column,
            #                                             str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(
            #                                                              monthTag + 'mess_all_' + op_column + '_' + op_type),
            #                                                          str(
            #                                                              monthTag + 'mess_all_' + op_column + '_' + op_type))
        ##################################################################
        #                 购买月份数(所有数据/30_101数据)
        ##################################################################
        op_type = 'nunique'
        op_column = 'month_order'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        ##################################################################
        #                 商品品类数(所有数据/30_101数据)
        ##################################################################
        op_type = 'nunique'
        op_column = 'cate'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_others, op_column,
                                                                 str(monthTag + op_column + '_other_' + op_type))
        ##################################################################
        #                 最近一次订单信息(所有数据/30_101数据)
        ##################################################################
        # OC_180slip_price_30_101_,OC_180slip_cate_30_101_,OC_180slip_para_1_30_101_,OC_180slip_para_2_30_101_,OC_180slip_para_3_30_101_,OC_180slip_o_area_30_101_,OC_180slip_o_sku_num_30_101_
        # OC_180slip_price_all_,OC_180slip_cate_all_,OC_180slip_para_1_all_,OC_180slip_para_2_all_,OC_180slip_para_3_all_,OC_180slip_o_area_all_,OC_180slip_o_sku_num_all_
        if j==180:
            op_column = 'o_date'
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Order_Sample_30_101, op_column,
                                                                     ['price', 'cate', 'para_1', 'para_2', 'para_3',
                                                                      'o_area', 'o_sku_num'],
                                                                     monthTag, '_30_101_')
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Order_Samples, op_column,
                                                                     ['price', 'cate', 'para_1', 'para_2', 'para_3',
                                                                      'o_area', 'o_sku_num'],
                                                                     monthTag, '_all_')
        ##################################################################
        #        判断最近一次订单是否为促销日(所有数据/30_101数据)
        ##################################################################
        sale_days = [datetime(2017, 6, 1), datetime(2017, 6, 2), datetime(2017, 6, 3), datetime(2017, 6, 4),
                     datetime(2017, 6, 5), datetime(2017, 6, 6), datetime(2017, 6, 7), datetime(2017, 6, 8),
                     datetime(2017, 6, 9), datetime(2017, 6, 10), datetime(2017, 6, 11), datetime(2017, 6, 12),
                     datetime(2017, 6, 13), datetime(2017, 6, 14), datetime(2017, 6, 15), datetime(2017, 6, 16),
                     datetime(2017, 6, 17), datetime(2017, 6, 18), datetime(2017, 6, 19), datetime(2017, 6, 20),
                     datetime(2016, 11, 11), datetime(2016, 12, 12)]
        if j == 180:
            op_column = 'o_date'
            self.Sample_data = eval('get_util.feature_%s' % 'last_action_sale') \
                (self.Sample_data, Order_Samples, op_column, str(monthTag + op_column + "_sale_all_"), sale_days)
            self.Sample_data = eval('get_util.feature_%s' % 'last_action_sale') \
                (self.Sample_data, Order_Sample_30_101, op_column, str(monthTag + op_column + "_sale_30_101_"),
                 sale_days)
        ##################################################################
        #          评论时间与对应订单时间的时间差(所有数据/30_101数据)
        ##################################################################
        op_column1 = 'o_date'
        op_column2 = 'comment_create_tm'
        df = eval('get_util.feature_%s' % 'time_gap')(Order_Samples, op_column2, op_column1,
                                                                    str(monthTag+ 'gap' + '_all_'+op_column1+"_"+op_column2))
        op_column = str(monthTag+ 'gap' + '_all_'+op_column1+"_"+op_column2)
        op_type = 'min'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(monthTag + op_column1+"_"+op_column2 + '_all_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(monthTag + op_column1 + "_" + op_column2 + '_all_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(monthTag + op_column1 + "_" + op_column2 + '_all_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(monthTag + op_column1 + "_" + op_column2 + '_all_' + op_type))

        df = eval('get_util.feature_%s' % 'time_gap')(Order_Sample_30_101, op_column2, op_column1,
                                                      str(monthTag + 'gap' + '_30_101_' + op_column1 + "_" + op_column2))
        op_column = str(monthTag + 'gap' + '_30_101_' + op_column1 + "_" + op_column2)
        op_type = 'min'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(
                                                                     monthTag + op_column1 + "_" + op_column2 + '_30_101_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(
                                                                     monthTag + op_column1 + "_" + op_column2 + '_30_101_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(
                                                                     monthTag + op_column1 + "_" + op_column2 + '_30_101_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df, op_column,
                                                                 str(
                                                                     monthTag + op_column1 + "_" + op_column2 + '_30_101_' + op_type))
        # ##################################################################
        # #                 商品复购率(30品类/101品类)
        # ##################################################################
        # # 统计时间段中购买品类30和101的购买商品复购率
        # op_column = 'o_sku_num'
        # self.Sample_data = eval('get_util.feature_%s' % 'buy_second_rate')(self.Sample_data, Order_Sample_30, op_column,
        #                                                          str(monthTag + op_column + '_30_second_' + op_type))
        # self.Sample_data = eval('get_util.feature_%s' % 'buy_second_rate')(self.Sample_data, Order_Sample_101, op_column,
        #                                                          str(monthTag + op_column + '_101_second_' + op_type))
        ##################################################################
        #         最大购买量到时间节点的时长(所有数据/30_101数据)
        ##################################################################
        # 'OC_30slip_mess_max_sku_all_min', 'OC_30slip_mess_max_sku_30_101_min', 'OC_60slip_mess_max_sku_all_min', 'OC_60slip_mess_max_sku_30_101_min', 'OC_90slip_mess_max_sku_all_min', 'OC_90slip_mess_max_sku_30_101_min', 'OC_180slip_mess_max_sku_all_min', 'OC_180slip_mess_max_sku_30_101_min', 'OC_15slip_mess_max_sku_all_min', 'OC_15slip_mess_max_sku_30_101_min'
        op_column = 'o_date'
        op_type = 'min'
        timePoint = datetime.strptime(label_date[0], '%Y-%m-%d')
        tmp = Order_Samples.sort_values(['user_id', 'o_sku_num'], ascending = False).drop_duplicates('user_id')\
            [['user_id', op_column]].\
            rename(columns={'user_id': 'user_id', op_column: 'o_date_max_sku'})
        df = eval('get_util.feature_%s' % 'op_gap')(tmp, 'o_date_max_sku', str(monthTag + 'mess_max_sku_all_' + op_type), timePoint)
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                 str(monthTag + 'mess_max_sku_all_' + op_type),
                                                                 str(monthTag + 'mess_max_sku_all_' + op_type))
        tmp = Order_Sample_30_101.sort_values(['user_id', 'o_sku_num'], ascending=False).drop_duplicates('user_id')\
            [['user_id', op_column]]. \
            rename(columns={'user_id': 'user_id', op_column: 'o_date_max_sku'})
        df = eval('get_util.feature_%s' % 'op_gap')(tmp, 'o_date_max_sku', str(monthTag + 'mess_max_sku_30_101_' + op_type), timePoint)
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                 str(monthTag + 'mess_max_sku_30_101_' + op_type),
                                                                 str(monthTag + 'mess_max_sku_30_101_' + op_type))
        ##################################################################
        #         用户购买日期为周末的比例(所有数据/30_101数据)
        ##################################################################
        # 'OC_30slip_rate_weekend_all_o_date_weekday', 'OC_30slip_rate_weekend_30_101_o_date_weekday', 'OC_60slip_rate_weekend_all_o_date_weekday', 'OC_60slip_rate_weekend_30_101_o_date_weekday', 'OC_90slip_rate_weekend_all_o_date_weekday', 'OC_90slip_rate_weekend_30_101_o_date_weekday','OC_180slip_rate_weekend_all_o_date_weekday', 'OC_180slip_rate_weekend_30_101_o_date_weekday','OC_15slip_rate_weekend_all_o_date_weekday', 'OC_15slip_rate_weekend_30_101_o_date_weekday'
        # weekday()方法可以返回星期数,0-6分别是周一到周日
        op_column = 'o_date_weekday'
        Order_Samples[op_column] = Order_Samples['o_date'].apply(lambda x:x.weekday())
        tmp = Order_Samples[['user_id',op_column]]
        tmp1 = tmp[(tmp[op_column] == 5) & (tmp[op_column] == 6)]# 周末的部分
        feature1 = tmp.groupby(['user_id'])[op_column].count().reset_index().\
            rename(columns={'user_id':'user_id', op_column:str(monthTag + 'full_all_' + op_column)})
        feature2 = tmp1.groupby(['user_id'])[op_column].count().reset_index().\
            rename(columns={'user_id':'user_id', op_column:str(monthTag + 'weekend_all_' + op_column)})
        feature = feature1.merge(feature2, on='user_id', how='left')
        feature[str(monthTag+'rate_weekend_all_'+op_column)] = feature[str(monthTag + 'weekend_all_' + op_column)]/feature[str(monthTag + 'full_all_' + op_column)]
        feature = feature[['user_id', str(monthTag+'rate_weekend_all_'+op_column)]]
        self.Sample_data = self.Sample_data.merge(feature, on='user_id', how='left')

        op_column = 'o_date_weekday'
        Order_Sample_30_101[op_column] = Order_Sample_30_101['o_date'].apply(lambda x: x.weekday())
        tmp = Order_Sample_30_101[['user_id', op_column]]
        tmp1 = tmp[(tmp[op_column] == 5) & (tmp[op_column] == 6)]  # 周末的部分
        feature1 = tmp.groupby(['user_id'])[op_column].count().reset_index(). \
            rename(columns={'user_id': 'user_id', op_column: str(monthTag + 'full_30_101_' + op_column)})
        feature2 = tmp1.groupby(['user_id'])[op_column].count().reset_index(). \
            rename(columns={'user_id': 'user_id', op_column: str(monthTag + 'weekend_30_101_' + op_column)})
        feature = feature1.merge(feature2, on='user_id', how='left')
        feature[str(monthTag + 'rate_weekend_30_101_' + op_column)] = feature[str(monthTag + 'weekend_30_101_' + op_column)] / \
                                                                   feature[str(monthTag + 'full_30_101_' + op_column)]
        feature = feature[['user_id', str(monthTag + 'rate_weekend_30_101_' + op_column)]]
        self.Sample_data = self.Sample_data.merge(feature, on='user_id', how='left')

        ##################################################################
        #        购买的天数(时间节点相关)(所有数据/30_101数据/30/101)
        ##################################################################
        op_type = 'min'
        op_column = 'day_order'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Order_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        ##################################################################
        #  用户购买日期为促销日的比例(所有数据/30_101数据/30数据/101数据)
        ##################################################################
        # 促销日
        sale_days = [datetime(2017, 6, 1), datetime(2017, 6, 2), datetime(2017, 6, 3), datetime(2017, 6, 4),
                    datetime(2017, 6, 5), datetime(2017,6,6), datetime(2017, 6, 7), datetime(2017, 6, 8),
                    datetime(2017, 6, 9), datetime(2017, 6, 10), datetime(2017, 6, 11), datetime(2017, 6, 12),
                    datetime(2017, 6, 13), datetime(2017, 6, 14), datetime(2017, 6, 15), datetime(2017, 6, 16),
                    datetime(2017, 6, 17), datetime(2017, 6, 18), datetime(2017, 6, 19), datetime(2017, 6, 20),
                    datetime(2016, 11, 11), datetime(2016, 12, 12)]
        op_column = 'o_id'
        self.Sample_data = eval('get_util.feature_%s' % 'order_sale_rate')(self.Sample_data, Order_Samples, op_column,
                                                                     str(monthTag + op_column + '_all_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'order_sale_rate')(self.Sample_data, Order_Sample_30_101, op_column,
                                                                     str(monthTag + op_column + '_30_101_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'order_sale_rate')(self.Sample_data, Order_Sample_30, op_column,
                                                                     str(monthTag + op_column + '_30_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'order_sale_rate')(self.Sample_data, Order_Sample_101, op_column,
                                                                     str(monthTag + op_column + '_101_' + 'rate'),
                                                                     sale_days)


    def FuncCommentFeatures(self, monthTag, sample_date, label_date, j):
        Comment_Samples = self.Order_Comment_User_Sku[ \
            (self.Order_Comment_User_Sku['comment_create_tm'] >= sample_date[0]) & \
            (self.Order_Comment_User_Sku['comment_create_tm'] <= sample_date[-1])]
        # 提取特征时间段的购买品类30和101的订单数据
        Comment_Sample_30_101 = Comment_Samples[(Comment_Samples['cate'] == 30) | (Comment_Samples['cate'] == 101)]
        Comment_Sample_30 = Comment_Samples[(Comment_Samples['cate'] == 30)]
        Comment_Sample_101 = Comment_Samples[(Comment_Samples['cate'] == 101)]

        ##################################################################
        #                用户评论分数(所有数据/30_101数据)
        ##################################################################
        op_column = 'score_level'
        op_type = 'count'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        op_type = 'sum'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        ##################################################################
        #   最近一次评论时间与时间节点之间的特征获取(所有数据/30_101数据)
        ##################################################################
        if j != 30:
            op_column = 'comment_create_tm'
            timePoint = datetime.strptime(label_date[0], '%Y-%m-%d')
            # 最小时间差
            op_type = 'min'
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 平均时间差
            op_type = 'mean'
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 方差
            op_type = 'std'
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                        timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Comment_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(
                                                                         monthTag + 'mess_all_' + op_column + '_' + op_type))
        ##################################################################
        #              最近一次评论信息(所有数据/30_101数据)
        ##################################################################
        # CC_180slip_price_30_101_,CC_180slip_cate_30_101_,CC_180slip_para_1_30_101_,CC_180slip_para_2_30_101_,CC_180slip_para_3_30_101_,CC_180slip_score_level_30_101_
        # CC_180slip_price_all_,CC_180slip_cate_all_,CC_180slip_para_1_all_,CC_180slip_para_2_all_,CC_180slip_para_3_all_,CC_180slip_score_level_all_
        if j==180:
            op_column = 'comment_create_tm'
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Comment_Sample_30_101, op_column,
                                                                          ['score_level', 'price', 'cate', 'para_1', 'para_2', 'para_3'],
                                                                          monthTag, '_30_101_')
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Comment_Samples, op_column,
                                                                          ['score_level', 'price', 'cate', 'para_1', 'para_2', 'para_3'],
                                                                          monthTag, '_all_')
        ##################################################################
        #                用户有无差评(所有数据/30_101数据)
        ##################################################################
        # 'CC_30slip_rate_poor_all_score_level', 'CC_60slip_rate_poor_all_score_level', 'CC_90slip_rate_poor_all_score_level', 'CC_180slip_rate_poor_all_score_level', 'CC_15slip_rate_poor_all_score_level'
        op_column = 'score_level'
        # # 有差评的用户
        # user_cha = Comment_Samples[Comment_Samples[op_column]==-3]. \
        #     drop_duplicates('user_id')[['user_id']]
        # user_cha[str(monthTag+'poor_all_'+op_column)] = 1
        # self.Sample_data = self.Sample_data.merge(user_cha, on='user_id', how='left')
        # self.Sample_data[str(monthTag+'poor_all_'+op_column)] = self.Sample_data[str(monthTag+'poor_all_'+op_column)].apply(lambda x:1 if x else 0)
        # 用户差评率
        cha = Comment_Samples[Comment_Samples[op_column] == 3].groupby(['user_id'])[op_column].count().\
            reset_index().rename(columns={'user_id':'user_id', op_column:'cha'})
        total = Comment_Samples.groupby(['user_id'])[op_column].count().reset_index(). \
            rename(columns={'user_id': 'user_id', op_column: 'total'})
        tmp = cha.merge(total, on=['user_id'], how='left')
        tmp[str(monthTag+'rate_poor_all_'+op_column)] = tmp['cha']/tmp['total']
        tmp = tmp[['user_id', str(monthTag+'rate_poor_all_'+op_column)]]
        self.Sample_data = self.Sample_data.merge(tmp, on='user_id', how='left')
        ##################################################################
        #        评论的天数(时间节点相关)(所有数据/30_101数据/30/101)
        ##################################################################
        op_type = 'min'
        op_column = 'day_comment'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Comment_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))

    def FuncActionFeatures(self, monthTag, sample_date, label_date, j):
        Action_Samples = self.Action_User_Sku[ \
            (self.Action_User_Sku['a_date'] >= sample_date[0]) & \
            (self.Action_User_Sku['a_date'] <= sample_date[-1])]
        Action_Sample_30_101 = Action_Samples[(Action_Samples['cate'] == 30) | (Action_Samples['cate'] == 101)]
        Action_Sample_30 = Action_Samples[(Action_Samples['cate'] == 30)]
        Action_Sample_101 = Action_Samples[(Action_Samples['cate'] == 101)]

        Action_Samples_look = Action_Samples[Action_Samples['a_type'] == 1]
        Action_Sample_30_101_look = Action_Sample_30_101[Action_Sample_30_101['a_type']==1]
        Action_Sample_30_look = Action_Sample_30[Action_Sample_30['a_type']==1]
        Action_Sample_101_look = Action_Sample_101[Action_Sample_101['a_type']==1]

        Action_Samples_star = Action_Samples[Action_Samples['a_type']==2]
        Action_Sample_30_101_star = Action_Sample_30_101[Action_Sample_30_101['a_type']==2]
        Action_Sample_30_star = Action_Sample_30[Action_Sample_30['a_type'] == 2]
        Action_Sample_101_star = Action_Sample_101[Action_Sample_101['a_type'] == 2]

        # 行为可分为1：浏览行为, 2：收藏行为, 3:全部
        ##################################################################
        #                行为天数(所有数据/30_101数据)
        ##################################################################
        op_type = 'nunique'
        op_column = 'a_date'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                               str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        ##################################################################
        #                商品种类数(所有数据/30_101数据)
        ##################################################################
        op_type = 'nunique'
        op_column = 'sku_id'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                               str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        ##################################################################
        #                总行为次数(所有数据/30_101数据)
        ##################################################################
        op_type = 'sum'
        op_column = 'a_num'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        ##################################################################
        #                浏览次数(所有数据/30_101数据)
        ##################################################################
        op_column = 'a_num'
        op_type = 'sum'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data,
                                                                 Action_Samples[Action_Samples['a_type'] == 1],
                                                                 op_column,
                                                                 str(monthTag + op_column + '_look_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data,
                                                                 Action_Sample_30_101[
                                                                     Action_Sample_30_101['a_type'] == 1],
                                                                 op_column,
                                                                 str(monthTag + op_column + '_look_30_101_' + op_type))
        ##################################################################
        #                收藏次数(所有数据/30_101数据)
        ##################################################################
        op_type = 'sum'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data,
                                                                 Action_Samples[Action_Samples['a_type'] == 2],
                                                                 op_column,
                                                                 str(monthTag + op_column + '_star_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data,
                                                                 Action_Sample_30_101[
                                                                     Action_Sample_30_101['a_type'] == 2],
                                                                 op_column,
                                                                 str(monthTag + op_column + '_star_30_101_' + op_type))
        op_column = 'cate'
        op_type = 'nunique'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                                 str(monthTag + op_column + '_30_101_' + op_type))
        if j != 30:
            ##################################################################
            #      最近一次操作距离时间节点的时间差(所有数据/30_101数据)
            ##################################################################
            op_column = 'a_date'
            # 最小时间差
            op_type = 'min'
            timePoint = datetime.strptime(label_date[0], '%Y-%m-%d')
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 平均时间差
            op_type = 'mean'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column  + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 方差
            op_type = 'std'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type))
            # 最大值
            op_type = 'max'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101, op_column,
                                                        str(monthTag + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples, op_column,
                                                        str(monthTag + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'mess_all_' + op_column + '_' + op_type))
            ##################################################################
            #      最近一次浏览距离时间节点的时间差(所有数据/30_101数据)
            ##################################################################
            op_column = 'a_date'
            op_type = 'min'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 1], op_column,
                                                        str(monthTag +  'look_' + 'mess_30_101_' + op_column + '_' + op_type ), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 1], op_column,
                                                        str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type))
            # 平均时间差
            op_type = 'mean'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 1],
                                                        op_column,
                                                        str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 1], op_column,
                                                        str(monthTag +  'look_' +'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type))
            # 方差
            op_type = 'std'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 1],
                                                        op_column,
                                                        str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 1], op_column,
                                                        str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type))
            # # 最大值
            # op_type = 'max'
            # df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 1],
            #                                             op_column,
            #                                             str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type),
            #                                                          str(monthTag + 'look_' + 'mess_30_101_' + op_column + '_' + op_type))
            # df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 1], op_column,
            #                                             str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type),
            #                                                          str(monthTag + 'look_' + 'mess_all_' + op_column + '_' + op_type))
            ##################################################################
            #      最近一次收藏距离时间节点的时间差(所有数据/30_101数据)
            ##################################################################
            op_column = 'a_date'
            op_type = 'min'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 2],
                                                        op_column,
                                                        str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 2], op_column,
                                                        str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type))
            # 平均时间差
            op_type = 'mean'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 2],
                                                        op_column,
                                                        str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 2], op_column,
                                                        str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type))
            # 方差
            op_type = 'std'
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 2],
                                                        op_column,
                                                        str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type))
            df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 2], op_column,
                                                        str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type),
                                                                     str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type))
            # # 最大值
            # op_type = 'max'
            # df = eval('get_util.feature_%s' % 'op_gap')(Action_Sample_30_101[Action_Sample_30_101['a_type'] == 2],
            #                                             op_column,
            #                                             str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type), timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type),
            #                                                          str(monthTag + 'star_' + 'mess_30_101_' + op_column + '_' + op_type))
            # df = eval('get_util.feature_%s' % 'op_gap')(Action_Samples[Action_Samples['a_type'] == 2], op_column,
            #                                             str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type), timePoint)
            # self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
            #                                                          str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type),
            #                                                          str(monthTag + 'star_' + 'mess_all_' + op_column + '_' + op_type))
        ##################################################################
        #              最近一次行为信息(所有数据/30_101数据)
        ##################################################################
        # AC_180slip_price_30_101_,AC_180slip_cate_30_101_,AC_180slip_para_1_30_101_,AC_180slip_para_2_30_101_,AC_180slip_para_3_30_101_,AC_180slip_a_num_30_101_,AC_180slip_a_type_30_101_
        # AC_180slip_price_all_,AC_180slip_cate_all_,AC_180slip_para_1_all_,AC_180slip_para_2_all_,AC_180slip_para_3_all_,AC_180slip_a_num_all_,AC_180slip_a_type_all_
        if j==180:
            # 最近一次行为信息
            op_column = 'a_date'
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Action_Sample_30_101,
                                                                          op_column,
                                                                          ['a_num', 'a_type', 'price', 'cate', 'para_1', 'para_2', 'para_3'],
                                                                          monthTag, '_30_101_')
            self.Sample_data = eval('get_util.feature_%s' % 'last_order')(self.Sample_data, Action_Samples, op_column,
                                                                          ['a_num', 'a_type', 'price', 'cate', 'para_1', 'para_2', 'para_3'],
                                                                          monthTag, '_all_')
        ##################################################################
        #        判断最近一次行为是否为促销日(所有数据/30_101数据)
        ##################################################################
        sale_days = [datetime(2017, 6, 1), datetime(2017, 6, 2), datetime(2017, 6, 3), datetime(2017, 6, 4),
                     datetime(2017, 6, 5), datetime(2017, 6, 6), datetime(2017, 6, 7), datetime(2017, 6, 8),
                     datetime(2017, 6, 9), datetime(2017, 6, 10), datetime(2017, 6, 11), datetime(2017, 6, 12),
                     datetime(2017, 6, 13), datetime(2017, 6, 14), datetime(2017, 6, 15), datetime(2017, 6, 16),
                     datetime(2017, 6, 17), datetime(2017, 6, 18), datetime(2017, 6, 19), datetime(2017, 6, 20),
                     datetime(2016, 11, 11), datetime(2016, 12, 12)]
        if j==180:
            op_column = 'a_date'
            self.Sample_data = eval('get_util.feature_%s' % 'last_action_sale')\
                (self.Sample_data, Action_Samples, op_column, str(monthTag + op_column + "_sale_all_"), sale_days)
            self.Sample_data = eval('get_util.feature_%s' % 'last_action_sale') \
                (self.Sample_data, Action_Sample_30_101, op_column, str(monthTag + op_column + "_sale_30_101_"), sale_days)
        ##################################################################
        #                点击率与转化率(所有数据/30_101数据)
        ##################################################################
        ## 所有数据
        Order_Samples = self.Order_Comment_User_Sku[ \
            (self.Order_Comment_User_Sku['o_date'] >= sample_date[0]) & \
            (self.Order_Comment_User_Sku['o_date'] <= sample_date[-1])]

        # 点击转化率=总订单数/浏览数
        # 统计总的订单数
        order_nums = Order_Samples.groupby(['user_id'])['o_id'].nunique().reset_index()
        # 统计总的浏览数
        action_look_nums = Action_Samples[Action_Samples['a_type'] == 1]. \
            groupby(['user_id'])['a_num'].sum().reset_index()
        feature = order_nums.merge(action_look_nums, on=['user_id'], how='inner')
        feature[str(monthTag + 'look_order' + '_all')] = feature['o_id'] / feature['a_num']
        feature = feature[['user_id', str(monthTag + 'look_order' + '_all')]]
        # 将大于1的转化率设置为1
        feature.ix[feature[str(monthTag + 'look_order' + '_all')] > 1.,
                   str(monthTag + 'look_order' + '_all')] = 1.
        self.Sample_data = self.Sample_data.merge(feature, on=['user_id'], how='left')
        # 收藏转化率=总订单数/收藏数
        # 统计总的订单数
        order_nums = Order_Samples.groupby(['user_id'])['o_id'].nunique().reset_index()
        # 统计总的浏览数
        action_star_nums = Action_Samples[Action_Samples['a_type'] == 2]. \
            groupby(['user_id'])['a_num'].sum().reset_index()
        feature = order_nums.merge(action_star_nums, on=['user_id'], how='inner')
        feature[str(monthTag + 'star_order' + '_all')] = feature['o_id'] / feature['a_num']
        feature = feature[['user_id', str(monthTag + 'star_order' + '_all')]]
        # 将大于1的转化率设置为1
        feature.ix[feature[str(monthTag + 'star_order' + '_all')] > 1.,
                   str(monthTag + 'star_order' + '_all')] = 1.
        self.Sample_data = self.Sample_data.merge(feature, on=['user_id'], how='left')

        ## 30_101数据
        # 提取特征时间段的购买品类30和101的订单数据
        Order_Sample_30_101 = Order_Samples[(Order_Samples['cate'] == 30) | (Order_Samples['cate'] == 101)]
        # 点击转化率=总订单数/浏览数
        # 统计总的订单数
        order_nums = Order_Sample_30_101.groupby(['user_id'])['o_id'].nunique().reset_index()
        # 统计总的浏览数
        action_look_nums = Action_Sample_30_101[Action_Sample_30_101['a_type'] == 1]. \
            groupby(['user_id'])['a_num'].sum().reset_index()
        feature = order_nums.merge(action_look_nums, on=['user_id'], how='inner')
        feature[str(monthTag + 'look_order' + '_30_101')] = feature['o_id'] / feature['a_num']
        feature = feature[['user_id', str(monthTag + 'look_order' + '_30_101')]]
        # 将大于1的转化率设置为1
        feature.ix[feature[str(monthTag + 'look_order' + '_30_101')] > 1.,
                   str(monthTag + 'look_order' + '_30_101')] = 1.
        self.Sample_data = self.Sample_data.merge(feature, on=['user_id'], how='left')
        # 收藏转化率=总订单数/收藏数
        # 统计总的订单数
        order_nums = Order_Sample_30_101.groupby(['user_id'])['o_id'].nunique().reset_index()
        # 统计总的浏览数
        action_star_nums = Action_Sample_30_101[Action_Sample_30_101['a_type'] == 2]. \
            groupby(['user_id'])['a_num'].sum().reset_index()
        feature = order_nums.merge(action_star_nums, on=['user_id'], how='inner')
        feature[str(monthTag + 'star_order' + '_30_101')] = feature['o_id'] / feature['a_num']
        feature = feature[['user_id', str(monthTag + 'star_order' + '_30_101')]]
        # 将大于1的转化率设置为1
        feature.ix[feature[str(monthTag + 'star_order' + '_30_101')] > 1.,
                   str(monthTag + 'star_order' + '_30_101')] = 1.
        self.Sample_data = self.Sample_data.merge(feature, on=['user_id'], how='left')
        ##################################################################
        #         最大行为到时间节点的时长(所有数据/30_101数据)
        ##################################################################
        # 'AC_30slip_mess_max_action_all_min', 'AC_30slip_mess_max_action_30_101_min', 'AC_60slip_mess_max_action_all_min', 'AC_60slip_mess_max_action_30_101_min', 'AC_90slip_mess_max_action_all_min', 'AC_90slip_mess_max_action_30_101_min', 'AC_180slip_mess_max_action_all_min', 'AC_180slip_mess_max_action_30_101_min' ,'AC_15slip_mess_max_action_all_min', 'AC_15slip_mess_max_action_30_101_min'
        op_column = 'a_date'
        op_type = 'min'
        timePoint = datetime.strptime(label_date[0], '%Y-%m-%d')
        tmp = Action_Samples.sort_values(['user_id', 'a_num'], ascending=False).drop_duplicates('user_id') \
            [['user_id', op_column]]. \
            rename(columns={'user_id': 'user_id', op_column: 'a_date_max_action'})
        df = eval('get_util.feature_%s' % 'op_gap')(tmp, 'a_date_max_action', str(monthTag + 'mess_max_action_all_' + op_type), timePoint)
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                 str(monthTag + 'mess_max_action_all_' + op_type),
                                                                 str(monthTag + 'mess_max_action_all_' + op_type))
        tmp = Action_Sample_30_101.sort_values(['user_id', 'a_num'], ascending=False).drop_duplicates('user_id') \
            [['user_id', op_column]]. \
            rename(columns={'user_id': 'user_id', op_column: 'a_date_max_action'})
        df = eval('get_util.feature_%s' % 'op_gap')(tmp, 'a_date_max_action', str(monthTag + 'mess_max_action_30_101_' + op_type), timePoint)
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, df,
                                                                 str(monthTag + 'mess_max_action_30_101_' + op_type),
                                                                 str(monthTag + 'mess_max_action_30_101_' + op_type))
        ##################################################################
        #        行为的天数(时间节点相关)(所有数据/30_101数据/30/101)
        ##################################################################
        op_type = 'min'
        op_column = 'day'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples, op_column,
                                                                 str(monthTag + op_column + '_all_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101, op_column,
                                                               str(monthTag + op_column + '_30_101_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30, op_column,
                                                                 str(monthTag + op_column + '_30_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_101, op_column,
                                                                 str(monthTag + op_column + '_101_' + op_type))
        ##################################################################
        #        浏览的天数(时间节点相关)(所有数据/30_101数据)
        ##################################################################
        op_type = 'min'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_look, op_column,
                                                                 str(monthTag + op_column + '_all_look_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_look, op_column,
                                                               str(monthTag + op_column + '_30_101_look_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_look, op_column,
                                                                 str(monthTag + op_column + '_all_look_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_look, op_column,
                                                               str(monthTag + op_column + '_30_101_look_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_look, op_column,
                                                                 str(monthTag + op_column + '_all_look_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_look, op_column,
                                                               str(monthTag + op_column + '_30_101_look_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_look, op_column,
                                                                 str(monthTag + op_column + '_all_look_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_look, op_column,
                                                               str(monthTag + op_column + '_30_101_look_' + op_type))
        ##################################################################
        #        收藏的天数(时间节点相关)(所有数据/30_101数据)
        ##################################################################
        op_type = 'min'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_star, op_column,
                                                                 str(monthTag + op_column + '_all_star_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_star, op_column,
                                                               str(monthTag + op_column + '_30_101_star_' + op_type))
        op_type = 'max'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_star, op_column,
                                                                 str(monthTag + op_column + '_all_star_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_star, op_column,
                                                               str(monthTag + op_column + '_30_101_star_' + op_type))
        op_type = 'mean'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_star, op_column,
                                                                 str(monthTag + op_column + '_all_star_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_star, op_column,
                                                               str(monthTag + op_column + '_30_101_star_' + op_type))
        op_type = 'std'
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Samples_star, op_column,
                                                                 str(monthTag + op_column + '_all_star_' + op_type))
        self.Sample_data = eval('get_util.feature_%s' % op_type)(self.Sample_data, Action_Sample_30_101_star, op_column,
                                                               str(monthTag + op_column + '_30_101_star_' + op_type))
        # ##################################################################
        # #      用户对该品类的行为数占总行为数的比例(30/101/30_101数据)
        # ##################################################################
        # op_column = 'a_date'
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #         (self.Sample_data, Action_Sample_30, Action_Samples, op_column,
        #          str(monthTag + op_column + '_rate_count_30'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data, Action_Sample_101, Action_Samples, op_column,
        #      str(monthTag + op_column + '_rate_count_101'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data, Action_Sample_30_101, Action_Samples, op_column,
        #      str(monthTag + op_column + '_rate_count_30_101'))
        # ##################################################################
        # #      用户对该品类的浏览数占总浏览数的比例(30/101/30_101数据)
        # ##################################################################
        # op_column = 'a_date'
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #         (self.Sample_data, Action_Sample_30[Action_Sample_30['a_type']==1],
        #          Action_Samples[Action_Samples['a_type']==1], op_column,
        #          str(monthTag + op_column + '_rate_count_30_look'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data, Action_Sample_101[Action_Sample_101['a_type']==1],
        #      Action_Samples[Action_Samples['a_type']==1], op_column,
        #      str(monthTag + op_column + '_rate_count_101_look'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data, Action_Sample_30_101[Action_Sample_30_101['a_type']==1],
        #       Action_Samples[Action_Samples['a_type']==1], op_column,
        #      str(monthTag + op_column + '_rate_count_30_101_look'))
        # ##################################################################
        # #      用户对该品类的收藏数占总收藏数的比例(30/101/30_101数据)
        # ##################################################################
        # op_column = 'a_date'
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #         (self.Sample_data, Action_Sample_30[Action_Sample_30['a_type']==2],
        #          Action_Samples[Action_Samples['a_type']==2], op_column,
        #          str(monthTag + op_column + '_rate_count_30_star'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data,  Action_Sample_101[Action_Sample_101['a_type']==2],
        #      Action_Samples[Action_Samples['a_type']==2], op_column,
        #      str(monthTag + op_column + '_rate_count_101_star'))
        # self.Sample_data = eval('get_util.feature_%s' % 'count_rate') \
        #     (self.Sample_data,  Action_Sample_30_101[Action_Sample_30_101['a_type']==2],
        #      Action_Samples[Action_Samples['a_type']==2], op_column,
        #      str(monthTag + op_column + '_rate_count_30_101_star'))
        ##################################################################
        #  用户行为日期为促销日的比例(所有数据/30_101数据/30数据/101数据)
        ##################################################################
        # 促销日
        sale_days = [datetime(2017, 6, 1), datetime(2017, 6, 2), datetime(2017, 6, 3), datetime(2017, 6, 4),
                     datetime(2017, 6, 5), datetime(2017, 6, 6), datetime(2017, 6, 7), datetime(2017, 6, 8),
                     datetime(2017, 6, 9), datetime(2017, 6, 10), datetime(2017, 6, 11), datetime(2017, 6, 12),
                     datetime(2017, 6, 13), datetime(2017, 6, 14), datetime(2017, 6, 15), datetime(2017, 6, 16),
                     datetime(2017, 6, 17), datetime(2017, 6, 18), datetime(2017, 6, 19), datetime(2017, 6, 20),
                     datetime(2016, 11, 11), datetime(2016, 12, 12)]
        op_column = 'a_num'
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Samples, op_column,
                                                                     str(monthTag + op_column + '_all_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30_101, op_column,
                                                                     str(monthTag + op_column + '_30_101_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30, op_column,
                                                                     str(monthTag + op_column + '_30_' + 'rate'),
                                                                     sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_101, op_column,
                                                                     str(monthTag + op_column + '_101_' + 'rate'),
                                                                     sale_days)
        ##################################################################
        #  用户浏览日期为促销日的比例(所有数据/30_101数据/30数据/101数据)
        ##################################################################
        op_column = 'a_num'
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Samples_look, op_column,
                                                                            str(monthTag + op_column + '_all_look_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30_101_look,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_30_101_look_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30_look,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_30_look_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_101_look,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_101_look_' + 'rate'),
                                                                            sale_days)
        ##################################################################
        #  用户关注日期为促销日的比例(所有数据/30_101数据/30数据/101数据)
        ##################################################################
        op_column = 'a_num'
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Samples_star,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_all_star_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30_101_star,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_30_101_star_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_30_star,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_30_star_' + 'rate'),
                                                                            sale_days)
        self.Sample_data = eval('get_util.feature_%s' % 'action_sale_rate')(self.Sample_data, Action_Sample_101_star,
                                                                            op_column,
                                                                            str(monthTag + op_column + '_101_star_' + 'rate'),
                                                                            sale_days)

    def FuncFeatureExamine(self, TagExamine):
        # 特征窗中最后一次行为时间
        # 特征为self.Sample_data, 判断其在Sample_30_101_LastTime日期上有没有订单行为
        # 即在self.Order_Comment_User_Sku上是否有对应的user_id与o_date与之对应
        self.Sample_data['o_date'] = self.Sample_data['Sample_30_101_LastTime']
        order = self.Order_Comment_User_Sku[['user_id', 'o_date']].sort_values(['user_id', 'o_date']). \
                drop_duplicates('user_id', keep='first')
        tmp = self.Sample_data.merge(order,
                                     on=['user_id', 'o_date'], how='inner')[['user_id', 'o_date']]. \
                                     rename(columns={'user_id': 'user_id', 'o_date': str(TagExamine+'Buy')})
        self.Sample_data = self.Sample_data.merge(tmp, on=['user_id'], how='left')
        self.Sample_data = self.Sample_data.drop(['o_date'], axis=1)
        self.Sample_data[str(TagExamine + 'Buy')] = \
            self.Sample_data[str(TagExamine + 'Buy')]. \
                apply(lambda x: 1 if str(pd.to_datetime(x))!='NaT' else 0)

    def FuncOnehotFeatures(self, data):
        # f_cols = ['sex', 'user_lv_cd', 'age']
        sex_df = pd.get_dummies(data['sex'], prefix='sex')
        user_lv_df = pd.get_dummies(data['user_lv_cd'], prefix='user_lv_cd')
        age_df = pd.get_dummies(data['age'], prefix='age')
        feature_col = [col for col in data.columns if
                       col not in ['sex', 'user_lv_cd', 'age']]
        data1 = pd.concat([data[feature_col], age_df, sex_df, user_lv_df], axis=1)
        return data1

    def FuncNaNcount(self, data):
        # 特征数+2个标签+user_id
        data['featurecount'] = data.shape[1]
        data['noNaNcount'] = data.count(axis=1)
        data['NaNcount'] = data['featurecount'] - data['noNaNcount']
        data = data.drop(['featurecount'], axis=1)
        data = data.drop(['noNaNcount'], axis=1)
        return data
