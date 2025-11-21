# airbnb_predict.py
# - 启动时检查本地 models/ 是否存在
# - 如果不存在：从外部网盘下载 models.zip，解压到 models/
# - 再加载每个城市的模型做预测

from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import requests
import zipfile

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_ZIP = BASE_DIR / "models.zip"
META_PATH = BASE_DIR / "airbnb_metadata.pkl"

# ★★★ 这里换成你在 Dropbox（或其它网盘）的直链地址 ★★★
MODEL_ZIP_URL = "https://www.dropbox.com/scl/fi/vhqlelhmwnj7rph5f57aj/models.zip?rlkey=g8gx4361nlesmc6asjgpd9oq7&st=myvq3uks&dl=1"



# ---------- 下载 & 解压 ----------

def _download_file(url: str, dest: Path) -> None:
    """从给定 URL 下载到本地 dest。"""
    print(f"[airbnb_predict] Downloading models.zip from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)
    print("[airbnb_predict] Download complete.")


def download_models_if_missing() -> None:
    """如果没有本地 models/，就从外部网盘下载并解压 models.zip。"""
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.pkl")):
        print("[airbnb_predict] models/ already exists, skip download.")
        return

    # 1. 下载 zip
    _download_file(MODEL_ZIP_URL, MODELS_ZIP)

    # 2. 检查是不是合法 zip（前两个字节应该是 PK）
    with open(MODELS_ZIP, "rb") as f:
        sig = f.read(4)
    if not sig.startswith(b"PK"):
        raise RuntimeError(
            "Downloaded file is NOT a valid ZIP. "
            "Please check the MODEL_ZIP_URL (it might be an HTML page)."
        )

    # 3. 解压
    print("[airbnb_predict] Extracting models.zip ...")
    with zipfile.ZipFile(MODELS_ZIP, "r") as zf:
        zf.extractall(BASE_DIR)
    print("[airbnb_predict] Extraction done.")

    # 4. 删除 zip
    MODELS_ZIP.unlink(missing_ok=True)


# ---------- 加载元数据 & 模型 ----------

download_models_if_missing()

if not META_PATH.exists():
    raise FileNotFoundError(f"Missing metadata file: {META_PATH}")

metadata = joblib.load(META_PATH)
feature_columns = metadata["feature_columns"]      # dict: city_key -> [col ...]
feature_medians = metadata["feature_medians"]      # dict: city_key -> {col: median}

city_model_paths = {p.stem: p for p in MODELS_DIR.glob("*.pkl")}
if not city_model_paths:
    raise RuntimeError(f"No model files found in {MODELS_DIR}")

artifacts = {
    "feature_columns": feature_columns,
    "feature_medians": feature_medians,
    "cities": sorted(city_model_paths.keys()),
}


def normalize_city_name(city: str) -> str:
    """把用户输入的城市名映射到模型文件的 key。"""
    if city is None:
        raise ValueError("City must not be None.")

    s = city.strip()

    # 1) 如果刚好就是 key，直接返回
    if s in city_model_paths:
        return s

    # 2) 否则做宽松匹配：忽略空格和下划线
    s_clean = s.lower().replace(" ", "").replace("_", "")
    for key in city_model_paths.keys():
        k_clean = key.lower().replace("_", "").replace(" ", "")
        if s_clean == k_clean:
            return key

    raise ValueError(
        f"Unsupported city '{city}'. Available: {sorted(city_model_paths.keys())}"
    )



@lru_cache(maxsize=None)
def get_city_model(city_key: str):
    """懒加载某个城市的模型（dict: {'xgb', 'rf', 'meta'}）。"""
    return joblib.load(city_model_paths[city_key])


def _stacked_predict(model_obj, X_df: pd.DataFrame) -> float:
    """stacked ensemble: xgb + rf -> meta regression，返回 log(price)。"""
    xgb = model_obj["xgb"]
    rf = model_obj["rf"]
    meta = model_obj["meta"]

    xgb_p = xgb.predict(X_df)
    rf_p = rf.predict(X_df)

    meta_in = np.column_stack([xgb_p, rf_p])
    y_log = meta.predict(meta_in)[0]
    return float(y_log)


def predict_price(city: str, features: dict) -> float:
    """主预测函数：返回真实价格（从 log 还原）。"""
    city_key = normalize_city_name(city)
    cols = feature_columns[city_key]
    med = feature_medians[city_key]

    # 按城市的列顺序组装一行特征，缺的用中位数补齐
    row = {c: features.get(c, med.get(c, 0)) for c in cols}

    X_df = pd.DataFrame([row], columns=cols)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(pd.Series(med)).fillna(0)

    model_obj = get_city_model(city_key)
    y_log = _stacked_predict(model_obj, X_df)
    return float(np.exp(y_log))


if __name__ == "__main__":
    print("Cities:", artifacts["cities"])
    test_city = artifacts["cities"][0]
    print("Test city:", test_city)
    print("Predicted price:", predict_price(test_city, {}))
