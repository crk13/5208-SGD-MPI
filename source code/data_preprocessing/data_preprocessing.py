import pandas as pd
import numpy as np
file_path = 'data/raw/nytaxi2022.csv'

# 取前10万行作为样本
df_sample = pd.read_csv(file_path, nrows=100000)

# #查看基本信息
# print("数据样本信息：")
# df_sample.info()

# print("\n前5行预览：")
# print(df_sample.head())

# ... 清洗代码 ...
# 处理重复和缺失值
df_sample = df_sample.drop_duplicates()
df_sample = df_sample.dropna()

#  问题1：将 object 类型转换为 datetime 类型
df_sample['tpep_pickup_datetime'] = pd.to_datetime(df_sample['tpep_pickup_datetime'])
df_sample['tpep_dropoff_datetime'] = pd.to_datetime(df_sample['tpep_dropoff_datetime'])
if df_sample['tpep_pickup_datetime'].isnull().any() or df_sample['tpep_dropoff_datetime'].isnull().any():
    print("⚠️ 警告：部分时间字符串格式有误，已转换为空值(NaT)。")
else:
    print("✔️ 时间列已成功转换为datetime类型。")

#问题2：乘客为0或负数
zero_passenger_count = df_sample[df_sample['passenger_count'] == 0].shape[0]
if zero_passenger_count > 0:
    print(f" 发现 {zero_passenger_count} 条记录的乘客数量为 0。")
# 删除乘客数为0的记录
df_sample = df_sample[df_sample['passenger_count'] != 0]

# 问题3：行程距离为0或负数
invalid_distance_count = df_sample[df_sample['trip_distance'] <= 0].shape[0]
if invalid_distance_count > 0:
    print(f" 发现 {invalid_distance_count} 条记录的行程距离为 0 或负数。")
# 删除相应记录
df_sample = df_sample[df_sample['trip_distance'] > 0]

# 问题4：总金额为负数或0
invalid_amount_count = df_sample[df_sample['total_amount'] <= 0].shape[0]
if invalid_amount_count > 0:
    print(f" 发现 {invalid_amount_count} 条记录的总金额为 0 或负数。")
# 删除相应记录
df_sample = df_sample[df_sample['total_amount'] > 0]

# 问题5： 逻辑一致性检查：下车时间早于或等于上车时间
# 计算行程时长（单位：分钟）
df_sample['trip_duration'] = (df_sample['tpep_dropoff_datetime'] - df_sample['tpep_pickup_datetime']).dt.total_seconds() / 60
invalid_duration_count = df_sample[df_sample['trip_duration'] <= 0].shape[0]
if invalid_duration_count > 0:
    print(f"🕵️ 发现 {invalid_duration_count} 条记录的行程时长为 0 或负数（下车时间早于上车时间）。")
# 删除行程时长为0或负数的记录
df_sample = df_sample[df_sample['trip_duration'] > 0]

# 问题6：store_and_fwd_flag bool转数值（Y/N转1/0）
df_sample['store_and_fwd_flag'] = df_sample['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
print("\n✅ 所有检查已完成！")


# 保存清洗后的样本数据
df_sample.to_csv('data/processed/sample_cleaned.csv',index=False)
# print("\n清洗后的数据样本信息：")
# df_sample.info()
# print("\n前5行预览：")
# print(df_sample.head())
