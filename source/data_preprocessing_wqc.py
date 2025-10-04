# python -m source.data_preprocessing_wqc \
#   --csv data/raw/nytaxi2022.csv \
#   --nrows None \
#   --out data/processed \
#   --chunksize 500000

import os
import argparse
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from pandas.api.types import CategoricalDtype
from scipy import sparse
from typing import Dict


# ------------------------
# 统一的预处理脚本（单进程）
# 读 CSV -> 清洗/特征工程 -> 切分 7:3 -> 保存为 npz（X, y）
# ------------------------

FEATURE_COLS = [
    'tpep_pickup_datetime',
    'tpep_dropoff_datetime',
    'passenger_count',
    'trip_distance',
    'RatecodeID',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'extra',
]
TARGET_COL = 'total_amount'

# 类别型取值（显式声明，避免 one-hot 时出现不一致列）
RATECODE_CATS = [1, 2, 3, 4, 5, 6, 99]
PAYMENT_CATS = [1, 2, 3, 4]
LOC_MIN, LOC_MAX = 1, 265  # PULocationID / DOLocationID 的取值范围

class AdaptiveHashBinner:
    """
    按类别频次自适应分桶（高频更细=每桶装更少类别，低频更粗）。
    - n_buckets: 最终桶数（含 unknown 桶）
    - dense_group_size: 高频区每桶容纳的类别数（越小越“细”）
    - sparse_group_size: 低频区每桶容纳的类别数（越大越“粗”）
    - use_signed: 是否用带符号哈希（+1/-1），减少碰撞偏向
    """
    def __init__(
        self,
        n_buckets: int = 20,
        dense_group_size: int = 10,
        sparse_group_size: int = 30,
        use_signed: bool = True,
        random_seed: int = 17,
        reserve_unknown: bool = True
    ):
        assert n_buckets >= 2, "n_buckets 至少为 2"
        self.n_buckets = n_buckets
        self.dense_group_size = dense_group_size
        self.sparse_group_size = sparse_group_size
        self.use_signed = use_signed
        self.random_seed = random_seed
        self.reserve_unknown = reserve_unknown

        self.bucket_map_: Dict[str, int] = {}
        self.unknown_bucket_: int = n_buckets - 1 if reserve_unknown else None
        self.info_: Dict[str, int] = {}  # 记录分桶信息

    @staticmethod
    def _signed_hash(x: str, seed: int) -> int:
        # Python 内置 hash 与进程相关，这里基于 tuple 固定化；仅用于符号，非索引
        h = hash((x, seed))
        return 1 if (h & (1 << 31)) == 0 else -1

    def fit(self, s: pd.Series):
        """根据频次把类别映射到 bucket_id。"""
        s = s.astype(str)
        vc = s.value_counts(dropna=False)  # 包含 'nan' 字符串
        cats = vc.index.tolist()
        n_unique = len(cats)

        # 预留 unknown 桶
        total_buckets = self.n_buckets - (1 if self.reserve_unknown else 0)
        if total_buckets <= 0:
            raise ValueError("桶数太少，无法预留 unknown。")

        # 计算应给“高频区”的桶数 k，使得：k*dense + (total_buckets-k)*sparse >= n_unique
        k_best = 0
        for k in range(total_buckets, -1, -1):  # 尽量多给高频
            cap = k * self.dense_group_size + (total_buckets - k) * self.sparse_group_size
            if cap >= n_unique:
                k_best = k
                break
        # 如果仍不足，则把 sparse_group_size 动态放大到刚好能容纳
        if k_best == 0:
            # 最小需要的 sparse 容量
            need = int(np.ceil(n_unique / total_buckets))
            if need > self.sparse_group_size:
                self.sparse_group_size = need

        k = k_best
        # 分配桶区间： [0 .. total_buckets-1] 用于已知类别；unknown（若有）用最后一个
        known_bucket_ids = list(range(total_buckets))

        # 把类别按频次顺序装桶
        ptr = 0
        bucket_idx = 0
        # 先装高频区（k 个桶，每桶 dense_group_size 类）
        for _ in range(k):
            cap = self.dense_group_size
            for _ in range(cap):
                if ptr >= n_unique: break
                cat = cats[ptr]
                self.bucket_map_[cat] = known_bucket_ids[bucket_idx]
                ptr += 1
            bucket_idx += 1
        # 再装低频区（剩余桶，每桶 sparse_group_size 类）
        while bucket_idx < total_buckets and ptr < n_unique:
            cap = self.sparse_group_size
            for _ in range(cap):
                if ptr >= n_unique: break
                cat = cats[ptr]
                self.bucket_map_[cat] = known_bucket_ids[bucket_idx]
                ptr += 1
            bucket_idx += 1

        # 若还有没装完（极端情况），直接循环塞最后一个已知桶
        while ptr < n_unique:
            self.bucket_map_[cats[ptr]] = known_bucket_ids[-1]
            ptr += 1

        # 记录信息
        self.info_ = {
            "n_unique": n_unique,
            "n_buckets_effective": total_buckets,
            "dense_buckets": k,
            "dense_group_size": self.dense_group_size,
            "sparse_buckets": total_buckets - k,
            "sparse_group_size": self.sparse_group_size,
            "has_unknown": int(self.reserve_unknown),
            "unknown_bucket": self.unknown_bucket_ if self.reserve_unknown else -1
        }
        return self

    def transform(self, s: pd.Series) -> sparse.csr_matrix:
        """把类别列转为 (n_samples, n_buckets) 的 CSR 稀疏矩阵（值= +1/-1）。"""
        if not self.bucket_map_:
            raise RuntimeError("请先 fit。")
        s = s.astype(str)
        n = len(s)
        rows, cols, data = [], [], []
        for i, val in enumerate(s.values):
            if val in self.bucket_map_:
                j = self.bucket_map_[val]
            else:
                if self.reserve_unknown:
                    j = self.unknown_bucket_
                else:
                    # 无 unknown 桶时，回退到 hash 到已知桶（更均衡）
                    j = hash((val, self.random_seed)) % self.n_buckets
            sign = self._signed_hash(val, self.random_seed) if self.use_signed else 1.0
            rows.append(i); cols.append(j); data.append(sign)
        X = sparse.csr_matrix((data, (rows, cols)), shape=(n, self.n_buckets), dtype=np.float32)
        return X

    def fit_transform(self, s: pd.Series) -> sparse.csr_matrix:
        return self.fit(s).transform(s)

    def get_bucket_map(self) -> Dict[str, int]:
        """返回 类别->桶 的映射字典。"""
        return dict(self.bucket_map_)

    def get_info(self) -> Dict[str, int]:
        """返回分桶配置与统计。"""
        return dict(self.info__)


def _log_step(tag: str, df: pd.DataFrame):
    print(f"[{tag}] 当前样本量: {len(df)} 行", flush=True)


def preprocess_csv(file_path: str,
                #    nrows: int | None = 100_000,
                   nrows: int | None = None,                   
                   output_dir: str = 'data/processed',
                   seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据：取指定列，包含目标列
    usecols = FEATURE_COLS + [TARGET_COL]
    print(f"开始读取 CSV：{file_path} ...", flush=True)
    if nrows is None:
        print("⚙️ 将读取整个文件（可能需要较长时间）", flush=True)
    else:
        print(f"⚙️ 将读取前 {nrows} 行", flush=True)

    start_t = time.time()
    chunksize = globals().get("__CHUNKSIZE__", None)  # 运行期通过 argparse 注入
    if chunksize is None:
        # 原有一次性读入
        df = pd.read_csv(file_path, nrows=nrows, usecols=usecols)
        print(f"✅ CSV 已读入：共 {len(df)} 行，{len(df.columns)} 列，用时 {time.time()-start_t:.1f}s", flush=True)
    else:
        # 分块流式读取并做最小清洗（去除关键列缺失），避免把无效行带入后续
        print(f"⚙️ 启用分块读取，每块 {chunksize} 行", flush=True)
        dfs = []
        read_rows = 0
        remaining = nrows  # 可能为 None
        for i, chunk in enumerate(pd.read_csv(file_path, usecols=usecols, chunksize=chunksize)):
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and chunk.shape[0] > remaining:
                chunk = chunk.iloc[:remaining]
            # 先丢掉关键列缺失，减少后续内存压力
            chunk = chunk.dropna(subset=usecols)
            # 固定格式解析时间
            for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
                chunk.loc[:, col] = pd.to_datetime(chunk[col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
            # 解析失败行丢弃
            before = chunk.shape[0]
            chunk = chunk.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
            # 块内提前做逻辑过滤：乘客>0、距离>0、时长>0（用 int64 纳秒差计算分钟）
            if not chunk.empty:
                chunk = chunk[chunk['passenger_count'] > 0]
                chunk = chunk[chunk['trip_distance'] > 0]
                if not chunk.empty:
                    # dtype-safe: ensure datetime64[ns] ndarray, then view as int64 nanoseconds
                    arr_drop = chunk['tpep_dropoff_datetime'].to_numpy(dtype='datetime64[ns]')
                    arr_pick = chunk['tpep_pickup_datetime'].to_numpy(dtype='datetime64[ns]')
                    ts_drop = arr_drop.view('i8')
                    ts_pick = arr_pick.view('i8')
                    dur_min = (ts_drop - ts_pick) / 1e9 / 60.0
                    chunk = chunk[dur_min > 0]
            after = chunk.shape[0]

            dfs.append(chunk)
            read_rows += after
            if remaining is not None:
                remaining -= after
            # 进度日志：读取块序号、累计有效行、当前块丢弃量
            print(f"  · 块 {i+1}: 保留 {after:,} 行（丢弃 {before-after:,}），累计有效 {read_rows:,}", flush=True)
        if len(dfs) == 0:
            raise ValueError("CSV 读取后没有有效行（关键列均缺失）。请检查原始文件或列名。")
        df = pd.concat(dfs, axis=0, ignore_index=True)
        print(f"✅ CSV 分块合并完成：共 {len(df)} 行，{len(df.columns)} 列，用时 {time.time()-start_t:.1f}s", flush=True)
        if df.shape[0] == 0:
            raise ValueError("分块读取后无有效数据，请检查 CSV 列名及时间列格式。")
        print("示例时间：", df[['tpep_pickup_datetime','tpep_dropoff_datetime']].head(3).to_dict('records'))

    # ---------------- 基础清洗 ----------------
    if chunksize is None:
        # 一次性读入：这里执行完整清洗流程
        # 去重 + 缺失值
        df = df.drop_duplicates()
        _log_step("去重", df)
        df = df.dropna(subset=usecols)
        _log_step("去除关键列缺失", df)

        # 简化时间解析，固定格式
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            df.loc[:, col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

        df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        _log_step("时间解析后去空", df)
        if len(df) == 0:
            raise ValueError("在时间解析和基础过滤后没有剩余样本。请检查原始CSV的时间格式或减少过滤条件（例如暂时不要过滤 passenger_count/trip_distance）。")

        # 逻辑过滤：乘客数>0、距离>0、时长>0
        df = df[df['passenger_count'] > 0]
        _log_step("过滤乘客数>0", df)
        df = df[df['trip_distance'] > 0]
        _log_step("过滤距离>0", df)

        # 确保时间列为 pandas datetime64[ns] 类型（避免 .dt 访问器报错）
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if not np.issubdtype(df[col].dtype, np.datetime64):
                df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')
        # 再次去掉无法解析的
        df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

        # 使用 int64 纳秒时间戳直接计算分钟差，避免 .dt 总是创建临时对象，速度更快
        arr_drop = df['tpep_dropoff_datetime'].to_numpy(dtype='datetime64[ns]')
        arr_pick = df['tpep_pickup_datetime'].to_numpy(dtype='datetime64[ns]')
        ts_drop = arr_drop.view('i8')
        ts_pick = arr_pick.view('i8')
        trip_duration_min = (ts_drop - ts_pick) / 1e9 / 60.0
        df = df[trip_duration_min > 0]
        _log_step("过滤时长>0", df)
    else:
        # 分块模式：各块已完成关键过滤（缺失/时间解析/乘客>0/距离>0/时长>0）
        # 这里仅做一次全局去重，避免重复过滤，节省时间与内存
        df = df.drop_duplicates()
        _log_step("全局去重（分块模式）", df)
        # 保险起见，确保时间列 dtype 正确（理论上块内已是 datetime64[ns]）
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if not np.issubdtype(df[col].dtype, np.datetime64):
                df.loc[:, col] = pd.to_datetime(df[col], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')

    # ---------------- 特征工程 ----------------
    # # 时间转数值特征（Unix 秒）；dtype-safe：先转为 datetime64[ns] ndarray，再视图为 int64 纳秒
    # arr_pick = df['tpep_pickup_datetime'].to_numpy(dtype='datetime64[ns]').view('i8')
    # arr_drop = df['tpep_dropoff_datetime'].to_numpy(dtype='datetime64[ns]').view('i8')
    # df['pickup_ts'] = (arr_pick // 10**9).astype('int64')
    # df['dropoff_ts'] = (arr_drop // 10**9).astype('int64')

    #分类周期化特征（hour_of_day, day_of_week）
    # 确保是 datetime64[ns]
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # 直接用 .dt 提取
    hours  = df['tpep_pickup_datetime'].dt.hour
    days   = df['tpep_pickup_datetime'].dt.dayofweek   # 0=Mon..6=Sun

    # 周期化
    df['hour_sin']  = np.sin(2 * np.pi * hours / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hours / 24)
    df['day_sin']   = np.sin(2 * np.pi * days / 7)
    df['day_cos']   = np.cos(2 * np.pi * days / 7)

    # 行程时长（分钟）
    df['trip_duration_min'] = (
        (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    ).astype('float32')

    # 如存在旧列，安全删除
    df.drop(columns=['pickup_ts', 'dropoff_ts'], inplace=True, errors='ignore')

    # 生成时间戳后，尽早删除原始 datetime 列，降低内存占用
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # 明确类别型（仅对 RatecodeID / payment_type 做 one-hot）
    # —— PULocationID / DOLocationID 保留为整型索引，后续在模型中用 Embedding
    df['RatecodeID'] = df['RatecodeID'].astype(CategoricalDtype(categories=RATECODE_CATS, ordered=False))
    df['payment_type'] = df['payment_type'].astype(CategoricalDtype(categories=PAYMENT_CATS, ordered=False))


    # 不做emdedding就没必要了
    # # 站点 ID：转为整数索引，超范围/缺失记为 0（留作 "unknown/pad"）
    # df['PULocationID'] = pd.to_numeric(df['PULocationID'], errors='coerce')
    # df['DOLocationID'] = pd.to_numeric(df['DOLocationID'], errors='coerce')
    # df.loc[~df['PULocationID'].between(LOC_MIN, LOC_MAX), 'PULocationID'] = np.nan
    # df.loc[~df['DOLocationID'].between(LOC_MIN, LOC_MAX), 'DOLocationID'] = np.nan
    # # 映射到 [0..265]：0=unknown，1..265 为真实 ID
    # df['PULocationID'] = df['PULocationID'].fillna(0).astype('int32')
    # df['DOLocationID'] = df['DOLocationID'].fillna(0).astype('int32')

    # 仅对 RatecodeID / payment_type 做 one-hot
    df = pd.get_dummies(
        df,
        columns=['RatecodeID', 'payment_type'],
        prefix=['Ratecode', 'paytype'],
        dummy_na=False
    )
    print(f"（PULocationID/DOLocationID 暂时保留，Ratecode/payment 做 one-hot）：当前特征数 {df.shape[1]}", flush=True)

    # 对'PUL'，'DOL'做类别型分桶（按频次自适应分桶，见 AdaptiveHashBinner）
    binner = AdaptiveHashBinner(
        n_buckets=20,           # 最终 20 桶（含 unknown）
        dense_group_size=10,    # 高频：10 类/桶
        sparse_group_size=30,   # 低频：30 类/桶
        use_signed=True,        # 带符号哈希，建议开
        reserve_unknown=True    # 预留 unknown 桶（最后一个）
    )
    # ✅ 改成字符串 + NaN 替换
    df['PULocationID'] = df['PULocationID'].fillna('__UNK__').astype(str)
    df['DOLocationID'] = df['DOLocationID'].fillna('__UNK__').astype(str)

    # 使用 AdaptiveHashBinner 转换成稀疏特征矩阵
    X_pu = binner.fit_transform(df['PULocationID'])
    X_do = binner.fit_transform(df['DOLocationID'])

    # 转回 DataFrame 并拼接
    pu_cols = [f'PULoc_bin_{i}' for i in range(binner.n_buckets)]
    do_cols = [f'DOLoc_bin_{i}' for i in range(binner.n_buckets)]
    df_pu = pd.DataFrame.sparse.from_spmatrix(X_pu, columns=pu_cols, index=df.index)
    df_do = pd.DataFrame.sparse.from_spmatrix(X_do, columns=do_cols, index=df.index)
    df = pd.concat([df, df_pu, df_do], axis=1)
    # 删除原始列
    df = df.drop(columns=['PULocationID', 'DOLocationID'])
    print(f"类别型特征分桶完成，当前特征数 {df.shape[1]}", flush=True)

    # 数值列归一化（不动 one-hot 列）
    num_cols = ['passenger_count', 'trip_distance', 'extra', 'trip_duration_min']
    for col in num_cols:
        if col not in df.columns:
            raise KeyError(f"数值列 {col} 缺失，请检查前置步骤。")
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("数值列归一化完成", flush=True)

    # 目标列
    y = df[[TARGET_COL]].astype(np.float32)

    # 组装最终特征矩阵：移除与目标列
    drop_cols = [TARGET_COL]
    X = df.drop(columns=drop_cols, errors='ignore').astype(np.float32)
    print(f"清洗后的最终样本量: {len(X)}，特征数: {X.shape[1]}", flush=True)

    # 打乱并 7:3 切分
    Xy = pd.concat([X, y], axis=1).sample(frac=1.0, random_state=seed)
    split_idx = int(0.7 * len(Xy))
    X_train = Xy.iloc[:split_idx, :-1].to_numpy()
    y_train = Xy.iloc[:split_idx,  -1].to_numpy().reshape(-1, 1)
    X_test  = Xy.iloc[split_idx:, :-1].to_numpy()
    y_test  = Xy.iloc[split_idx:,  -1].to_numpy().reshape(-1, 1)

    # 保存为 npz（训练端直接读取 X, y）
    feature_names = np.array(X.columns.tolist(), dtype=object)
    np.savez(os.path.join(output_dir, 'train.npz'), X=X_train, y=y_train, feature_names=feature_names)
    np.savez(os.path.join(output_dir, 'test.npz'),  X=X_test,  y=y_test,  feature_names=feature_names)

    # 也保存一个清洗后的样本 CSV（便于查看，非必须）
    sample_preview = os.path.join(output_dir, 'sample_cleaned.csv')
    Xy.head(1000).to_csv(sample_preview, index=False)

    print(f"预处理完成：X_train={X_train.shape}, X_test={X_test.shape}，特征数={feature_names.size}")
    print(f"已保存：{os.path.join(output_dir, 'train.npz')} / {os.path.join(output_dir, 'test.npz')}")
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/raw/nytaxi2022.csv')
    parser.add_argument(
        '--nrows',
        type=lambda x: None if str(x).lower() == "none" else int(x),
        default=100_0000,
        help="行数 (int)；传入 None 表示读取整个文件"
    )
    parser.add_argument('--out', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunksize', type=lambda x: None if str(x).lower()=="none" else int(x), default=None,
                        help="分块读取行数（None=一次性读入；建议大文件用 200000~1000000）")
    args = parser.parse_args()

    # 将 chunksize 注入到模块级变量，供 preprocess_csv 使用
    globals()['__CHUNKSIZE__'] = args.chunksize

    preprocess_csv(args.csv, nrows=args.nrows, output_dir=args.out, seed=args.seed)


if __name__ == '__main__':
    main()
