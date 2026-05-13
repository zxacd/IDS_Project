import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import joblib
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_MAP_SIZE = (32, 32)
SAMPLE_FRAC = 1.0   # 使用全部数据，保证教师和学生模型的知识质量
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

def merge_csv_files():
    all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"未找到CSV文件: {RAW_DATA_DIR}")
    logger.info(f"找到 {len(all_files)} 个CSV文件")
    df_list = []
    for f in all_files:
        logger.info(f"读取 {os.path.basename(f)}")
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
        except:
            df = pd.read_csv(f, encoding='latin1', low_memory=False)
        df_list.append(df)
    total = pd.concat(df_list, ignore_index=True)
    logger.info(f"合并完成: {total.shape}")
    return total

def clean_data(df):
    df = df.dropna(axis=1, how='all')
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median if not pd.isna(median) else 0, inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col].fillna(mode[0] if len(mode) > 0 else 'unknown', inplace=True)

    df = df.drop_duplicates()
    logger.info(f"清洗后数据形状: {df.shape}")
    return df

def process_features(df):
    label_col = None
    for col in df.columns:
        if 'label' in col.lower() or 'attack' in col.lower():
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]

    # ======================
    # 【在这里过滤稀有类别】
    # ======================
    cnt = df[label_col].value_counts()
    valid_labels = cnt[cnt >= 2].index
    df = df[df[label_col].isin(valid_labels)]
    logger.info(f"过滤稀有类别后: {df.shape}")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols)

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    logger.info(f"特征数量: {X_scaled.shape[1]}, 类别数: {len(le.classes_)}")
    return X_scaled, y_enc, scaler, le, feature_names

def build_feature_maps(X_scaled):
    """
    极速构建特征图：直接 pad/truncate 到 1024 维 → reshape 32×32
    完全向量化，无 Python 循环、无 resize，比原实现快 50~100 倍
    """
    n_samples, n_features = X_scaled.shape
    target = FEATURE_MAP_SIZE[0] * FEATURE_MAP_SIZE[1]  # 32*32=1024

    if n_features >= target:
        # 特征过多，截断
        padded = X_scaled[:, :target].astype(np.float32)
    else:
        # 特征不足，向量化补零
        padded = np.zeros((n_samples, target), dtype=np.float32)
        padded[:, :n_features] = X_scaled

    # 一次性 reshape，无需循环、无需 resize
    maps = padded.reshape(n_samples, FEATURE_MAP_SIZE[0], FEATURE_MAP_SIZE[1])
    return maps[..., np.newaxis]

def main():
    merged_file = 'data/total.csv'

    if os.path.exists(merged_file):
        logger.info("检测到旧数据，删除并重新生成...")
        os.remove(merged_file)

    total_df = merge_csv_files()
    total_df.to_csv(merged_file, index=False)

    if SAMPLE_FRAC < 1.0:
        total_df = total_df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)

    total_df = clean_data(total_df)
    if total_df.shape[0] == 0:
        logger.error("清洗后数据为空，请检查原始数据或清洗逻辑")
        return

    X, y, scaler, le, feature_names = process_features(total_df)

    # ======================
    # 【安全划分，不会报错】
    # ======================
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=VAL_SIZE+TEST_SIZE, stratify=y, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), stratify=y_temp, random_state=RANDOM_SEED
        )
    except:
        logger.warning("分层划分失败，使用普通划分")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=VAL_SIZE+TEST_SIZE, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), random_state=RANDOM_SEED
        )

    X_train_img = build_feature_maps(X_train)
    X_val_img = build_feature_maps(X_val)
    X_test_img = build_feature_maps(X_test)

    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train_img)
    np.save(f"{PROCESSED_DIR}/X_val.npy", X_val_img)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test_img)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
    np.save(f"{PROCESSED_DIR}/y_val.npy", y_val)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")

    logger.info("✅ 预处理完成！")

if __name__ == '__main__':
    main()