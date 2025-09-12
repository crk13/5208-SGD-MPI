import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import CategoricalDtype

file_path = 'data/raw/nytaxi2022.csv'

# 取前10万行作为样本
df_sample = pd.read_csv(file_path, 
                        nrows=100000,
                        usecols=['tpep_pickup_datetime', 
                                'tpep_dropoff_datetime',
                                'passenger_count', 
                                'trip_distance',
                                'RatecodeID',
                                'PULocationID',
                                'DOLocationID',
                                'payment_type',
                                'extra'])
# #查看基本信息
# print("数据样本信息：")
# df_sample.info()

# print("\n前5行预览：")
# print(df_sample.head())

# ... 清洗代码 ...
# 处理重复和缺失值
df_sample = df_sample.drop_duplicates()
df_sample = df_sample.dropna()

# 1：将 object 类型转换为 datetime 类型
df_sample['tpep_pickup_datetime'] = pd.to_datetime(df_sample['tpep_pickup_datetime'])
df_sample['tpep_dropoff_datetime'] = pd.to_datetime(df_sample['tpep_dropoff_datetime'])
if df_sample['tpep_pickup_datetime'].isnull().any() or df_sample['tpep_dropoff_datetime'].isnull().any():
    print("⚠️ 警告：部分时间字符串格式有误，已转换为空值(NaT)。")
else:
    print("✔️ 时间列已成功转换为datetime类型。")

# 2：乘客为0或负数
zero_passenger_count = df_sample[df_sample['passenger_count'] == 0].shape[0]
if zero_passenger_count > 0:
    print(f" 发现 {zero_passenger_count} 条记录的乘客数量为 0。")
# 删除乘客数为0的记录
df_sample = df_sample[df_sample['passenger_count'] != 0]

# 3：行程距离为0或负数
invalid_distance_count = df_sample[df_sample['trip_distance'] <= 0].shape[0]
if invalid_distance_count > 0:
    print(f" 发现 {invalid_distance_count} 条记录的行程距离为 0 或负数。")
# 删除相应记录
df_sample = df_sample[df_sample['trip_distance'] > 0]


# 4：逻辑一致性检查：下车时间早于或等于上车时间
# 计算行程时长（单位：分钟）
df_sample['trip_duration'] = (df_sample['tpep_dropoff_datetime'] - df_sample['tpep_pickup_datetime']).dt.total_seconds() / 60
invalid_duration_count = df_sample[df_sample['trip_duration'] <= 0].shape[0]
if invalid_duration_count > 0:
    print(f"🕵️ 发现 {invalid_duration_count} 条记录的行程时长为 0 或负数（下车时间早于上车时间）。")
# 删除行程时长为0或负数的记录
df_sample = df_sample[df_sample['trip_duration'] > 0]
df_sample = df_sample.drop(columns=['trip_duration'])  # 清洗后删除辅助列

# 5：RatecodeID转为独热编码
rate_code_categories = [1, 2, 3, 4, 5, 6, 99]
rate_code_type = CategoricalDtype(categories=rate_code_categories, ordered=False)
df_sample['RatecodeID'] = df_sample['RatecodeID'].astype(rate_code_type)
df_sample = pd.get_dummies(df_sample, columns=['RatecodeID'], prefix='Ratecode')

# 6: Payment_type转为类别型
payment_type_categories = [1, 2, 3, 4]
payment_type_type = CategoricalDtype(categories=payment_type_categories, ordered=False)
df_sample['payment_type'] = df_sample['payment_type'].astype(payment_type_type)
df_sample = pd.get_dummies(df_sample, columns=['payment_type'], prefix='paytype')

# 选取数值型列
num_cols =['passenger_count', 'trip_distance', 'extra']

# 初始化归一化器
scaler = MinMaxScaler()

# 归一化并替换原数据
df_sample[num_cols] = scaler.fit_transform(df_sample[num_cols])

# 保存清洗后的样本数据
df_sample.to_csv('data/processed/sample_cleaned.csv',index=False)
print("\n清洗后的数据样本信息：")
df_sample.info()
print("\n前5行预览：")
print(df_sample.head())
