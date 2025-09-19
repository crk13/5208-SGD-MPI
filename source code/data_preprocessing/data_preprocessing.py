import pandas as pd
from chunk_cleaning import chunk_cleaning
from switch_class import switch_class
import os
import time

#1.配置参数
file_path = 'data/raw/nytaxi2022.csv'
processed_file_path = 'data/processed/nytaxi2022_cleaned_full.csv'
chunk_size = 1000000  # 每次处理100万行
columns_to_use=['tpep_pickup_datetime', 
               'tpep_dropoff_datetime',
                'passenger_count', 
                'trip_distance',
                'RatecodeID',
                'PULocationID',
                'DOLocationID',
                'payment_type',
                'extra',
                'total_amount']
#需要归一化的列
num_cols =['passenger_count', 'trip_distance', 'extra']

#2.计算全局 Min/Max (Stats Pass)
print("[Phase 1] Starting: Calculating global Min/Max for scaling...")
start_time = time.time()
global_min_max_stats = {col: {'min': float('inf'), 'max': float('-inf')} for col in num_cols}
chunk_iterator_stats = pd.read_csv(
    file_path, 
    chunksize=chunk_size, 
    usecols=columns_to_use,
    low_memory=False
)

for i, chunk in enumerate(chunk_iterator_stats):
    print(f"  - Scanning chunk {i+1} for Min/Max stats...")
    # 先进行过滤，以获得更准确的求min/max的范围
    chunk = chunk_cleaning(chunk)
    
    for col in num_cols:
        current_min = chunk[col].min()
        current_max = chunk[col].max()
        if current_min < global_min_max_stats[col]['min']:
            global_min_max_stats[col]['min'] = current_min
        if current_max > global_min_max_stats[col]['max']:
            global_min_max_stats[col]['max'] = current_max

phase1_time = time.time() - start_time
print(f"[Phase 1] Completed in {phase1_time:.2f} seconds. Global stats collected:")
print(global_min_max_stats)


print(f"\n [Phase 2] Starting: Processing full dataset and writing to '{processed_file_path}'...")
start_time = time.time()

if os.path.exists(processed_file_path):
    os.remove(processed_file_path)

chunk_iterator_process = pd.read_csv(
    file_path, 
    chunksize=chunk_size, 
    usecols=columns_to_use,
    low_memory=False
)
is_first_chunk = True

for i, chunk in enumerate(chunk_iterator_process):
    print(f"  - Processing and writing chunk {i+1}...")
    
    # 处理数据
    processed_chunk = chunk_cleaning(chunk)
    processed_chunk = switch_class(processed_chunk)

    # 进行 Min-Max 归一化
    for col in num_cols:
        min_val = global_min_max_stats[col]['min']
        max_val = global_min_max_stats[col]['max']
        if (max_val - min_val) > 0:
            processed_chunk[f'{col}_scaled'] = (processed_chunk[col] - min_val) / (max_val - min_val)
        else:
            processed_chunk[f'{col}_scaled'] = 0
    #归一化后删除原始列
    processed_chunk = processed_chunk.drop(columns=num_cols)

    # 写入文件
    processed_chunk.to_csv(processed_file_path, mode='a', header=is_first_chunk, index=False)
    is_first_chunk = False

phase2_time = time.time() - start_time
print(f"\n✅ [Phase 2] Completed in {phase2_time:.2f} seconds.")
print(f"All data has been processed. Clean data is saved to: '{processed_file_path}'")
