# airbnb_predict.py
# - 直接使用本地 models/<city>.pkl（不下载）
# - 从 airbnb_metadata.pkl 读特征 & 中位数
# - 每个城市是 stacked model: xgb + rf + linear meta
# - 输出的是还原后的价格（exp(log_price)）

from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 0. 路径 & 元数据
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

META_PATH = BASE_DIR / "airbnb_metadata.pkl"
MODELS_DIR = BASE_DIR / "models"

if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

if not MODELS_DIR.exists():
    raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

metadata = joblib.load(META_PATH)
feature_columns = metadata["feature_columns"]   # city_key -> [columns]
feature_medians = metadata["feature_medians"]   # city_key -> {col: median}

# 自动发现每个城市的模型文件
city_model_paths = {p.stem: p for p in MODELS_DIR.glob("*.pkl")}
if not city_model_paths:
    raise RuntimeError(f"No model files found in {MODELS_DIR}")

artifacts = {
    "feature_columns": feature_columns,
    "feature_medians": feature_medians,
    "cities": sorted(city_model_paths.keys()),
}

# -------------------------------------------------------------------
# 1. 工具函数
# -------------------------------------------------------------------

def normalize_city_name(city: str) -> str:
    """
    用户输入城市名 -> 内部 key（例如 'Los_Angeles'）
    忽略大小写、空格/下划线差异。
    """
    if city is None:
        raise ValueError("City must not be None.")

    s = city.strip().lower()
    s_nospace = s.replace(" ", "")
    s_us = s.replace(" ", "_")

    for key in city_model_paths.keys():
        k = key.lower()
        k_nospace = k.replace("_", "")
        if s == k or s_nospace == k_nospace or s_us == k:
            return key

    raise ValueError(
        f"City '{city}' is not supported. "
        f"Available cities: {sorted(city_model_paths.keys())}"
    )


@lru_cache(maxsize=None)
def get_city_model(city_key: str):
    """
    懒加载某个城市的模型：models/<city_key>.pkl
    文件内容是 dict:
      {'xgb': XGBRegressor, 'rf': RandomForestRegressor, 'meta': LinearRegression}
    """
    if city_key not in city_model_paths:
        raise KeyError(f"No model path recorded for city '{city_key}'")
    path = city_model_paths[city_key]
    return joblib.load(path)


def _stacked_predict(model_obj, X_df: pd.DataFrame) -> float:
    """
    跑 stacked ensemble，返回预测的 log-price。

    model_obj:
      - dict: {'xgb', 'rf', 'meta'}
      - 或者任何有 .predict() 的单一模型
    """
    if hasattr(model_obj, "predict"):
        return float(model_obj.predict(X_df)[0])

    if isinstance(model_obj, dict):
        keys = set(model_obj.keys())
        if {"xgb", "rf", "meta"} <= keys:
            xgb = model_obj["xgb"]
            rf = model_obj["rf"]
            meta = model_obj["meta"]

            xgb_pred = xgb.predict(X_df)
            rf_pred = rf.predict(X_df)

            meta_input = np.column_stack([xgb_pred, rf_pred])
            y_log = meta.predict(meta_input)[0]
            return float(y_log)

    raise ValueError("Unsupported model object structure for prediction.")


# -------------------------------------------------------------------
# 2. 对外接口：predict_price
# -------------------------------------------------------------------

def predict_price(city: str, features: dict) -> float:
    """
    给定城市 + 特征，预测每晚价格（美元）。

    city: 例如 "Los Angeles", "San Francisco"
    features: {特征名: 值}，缺失特征用 city 的中位数补。
    """
    if not isinstance(features, dict):
        raise ValueError("features must be a dict of {feature_name: value}.")

    city_key = normalize_city_name(city)

    cols = feature_columns[city_key]
    med = feature_medians[city_key]

    # 用中位数兜底
    row = {}
    for col in cols:
        if col in features:
            row[col] = features[col]
        else:
            row[col] = med.get(col, 0)

    X_df = pd.DataFrame([row], columns=cols)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(pd.Series(med)).fillna(0)

    model_obj = get_city_model(city_key)

    # 模型输出 log(price)，这里还原成 price
    pred_log = _stacked_predict(model_obj, X_df)
    price = float(np.exp(pred_log))
    return price


# -------------------------------------------------------------------
# 3. 本地测试
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:
        example_city = artifacts["cities"][0]
        print("Example city key:", example_city)
        print("Predicted price:", predict_price(example_city, {}))
    except Exception as e:
        print("Error during test prediction:", e)
