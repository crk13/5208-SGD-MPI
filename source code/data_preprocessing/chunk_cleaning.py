# 实现数据块清洗功能
import pandas as pd

def chunk_cleaning(chunk):
    # 处理重复和缺失值
    chunk = chunk.drop_duplicates()
    chunk = chunk.dropna()
    # 将 object 类型转换为 datetime 类型
    chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'], errors='coerce')
    chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'], errors='coerce')
    # 逻辑一致性检查：下车时间早于或等于上车时间
    chunk = chunk[chunk['tpep_dropoff_datetime'] > chunk['tpep_pickup_datetime']]
    # 删除乘客数<=0的记录
    chunk = chunk[chunk['passenger_count'] > 0]
    # 删除行程距离为0或负数的记录
    chunk = chunk[chunk['trip_distance'] > 0]
    # 删除extra列负值
    chunk = chunk[chunk['extra'] >= 0]
    return chunk 