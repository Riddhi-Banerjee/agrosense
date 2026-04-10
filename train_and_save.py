# ============================================================
# train_and_save.py
# Run this ONCE in Google Colab to train all models and save
# them as .pkl files. Upload those files to your GitHub repo.
#
# Usage:
#   python train_and_save.py
#   (or run each cell in Colab)
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_AVAILABLE = False

os.makedirs("models", exist_ok=True)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
# Update this path to your CSV location
file_path = "/content/drive/MyDrive/agriculture_new/Smart_Farming_Crop_Yield_2024.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.replace(" ", "_")
df = df.rename(columns={
    'Soil_Moisture_(%)': 'soil_moisture_%',
    'Soil_pH':           'soil_pH',
    'Temperature(C)':    'temperature_C',
    'Rainfall_(mm)':     'rainfall_mm',
    'Humidity_(%)':      'humidity_%',
    'Crop_Type':         'crop_type',
    'Region':            'region'
})

FEATURES = [
    'soil_moisture_%', 'soil_pH', 'temperature_C',
    'rainfall_mm', 'humidity_%', 'NDVI_index'
]
YIELD_FEATURES = FEATURES + [
    'sunlight_hours', 'pesticide_usage_ml',
    'total_days', 'season_sin', 'season_cos',
    'crop_type', 'region'
]
INJECTION_FRACTION = 0.10

# ============================================================
# 2. SEASONAL ENCODING
# ============================================================
df['sowing_date'] = pd.to_datetime(df['sowing_date'], dayfirst=True, errors='coerce')
df['sowing_month'] = df['sowing_date'].dt.month.fillna(6)
df['season_sin'] = np.sin(2 * np.pi * df['sowing_month'] / 12)
df['season_cos'] = np.cos(2 * np.pi * df['sowing_month'] / 12)

# ============================================================
# 3. ENCODING
# ============================================================
encoders = {}
for col in ['region', 'crop_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================================================
# 4. ANOMALY INJECTION
# ============================================================
def inject_contextual_anomalies(df, frac=INJECTION_FRACTION, seed=42):
    np.random.seed(seed)
    df = df.copy()
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'normal'
    for (crop, region), data in df.groupby(['crop_type', 'region']):
        if len(data) < 15:
            continue
        n = max(1, int(len(data) * frac))
        idx = np.random.choice(data.index, n, replace=False)
        seg_stats = {f: {'low': data[f].quantile(0.10),
                         'high': data[f].quantile(0.90)} for f in FEATURES}
        for i in idx:
            atype = np.random.choice(
                ['drought_stress', 'heat_stress', 'sensor_inconsistency', 'crop_failure'],
                p=[0.25, 0.25, 0.25, 0.25]
            )
            if atype == 'drought_stress':
                df.loc[i, 'soil_moisture_%'] = seg_stats['soil_moisture_%']['low'] * 0.20
                df.loc[i, 'rainfall_mm']      = seg_stats['rainfall_mm']['low']    * 0.30
                df.loc[i, 'NDVI_index']       = seg_stats['NDVI_index']['low']     * 0.40
            elif atype == 'heat_stress':
                df.loc[i, 'temperature_C']    = seg_stats['temperature_C']['high'] * 1.50
                df.loc[i, 'humidity_%']        = seg_stats['humidity_%']['high']    * 1.30
            elif atype == 'sensor_inconsistency':
                df.loc[i, 'rainfall_mm']      = seg_stats['rainfall_mm']['high']
                df.loc[i, 'soil_moisture_%']  = seg_stats['soil_moisture_%']['low'] * 0.20
            elif atype == 'crop_failure':
                df.loc[i, 'NDVI_index']       = seg_stats['NDVI_index']['low']     * 0.30
                df.loc[i, 'soil_moisture_%']  = seg_stats['soil_moisture_%']['low'] * 0.20
                df.loc[i, 'temperature_C']    = seg_stats['temperature_C']['high'] * 1.50
            df.loc[i, 'is_anomaly']   = 1
            df.loc[i, 'anomaly_type'] = atype
    print(f"Injected {df['is_anomaly'].sum()} anomalies ({100*df['is_anomaly'].mean():.1f}%)")
    return df

df_eval = inject_contextual_anomalies(df)

# ============================================================
# 5. TRAIN PER-SEGMENT MODELS
# ============================================================
if_models, lof_models, autoencoders, scalers, fusion_models = {}, {}, {}, {}, {}

# Store per-segment normal range for explanation in dashboard
segment_stats = {}

for (crop, region), data in df_eval.groupby(['crop_type', 'region']):
    if len(data) < 15:
        continue
    normal_data = data[data['is_anomaly'] == 0]
    X_normal    = normal_data[FEATURES].values

    # Normal range stats for dashboard display
    segment_stats[(crop, region)] = {
        f: {'low': normal_data[f].quantile(0.10),
            'high': normal_data[f].quantile(0.90),
            'mean': normal_data[f].mean()}
        for f in FEATURES
    }

    # Isolation Forest
    ifm = IsolationForest(contamination=INJECTION_FRACTION, random_state=42)
    ifm.fit(X_normal)
    if_models[(crop, region)] = ifm

    # LOF
    n_nbrs = min(20, max(5, len(X_normal) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_nbrs,
                             contamination=INJECTION_FRACTION, novelty=True)
    lof.fit(X_normal)
    lof_models[(crop, region)] = lof

    # Autoencoder
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_normal)
    scalers[(crop, region)] = scaler

    inp = Input(shape=(X_normal.shape[1],))
    x   = Dense(8, activation='relu')(inp)
    x   = Dense(4, activation='relu')(x)
    x   = Dense(8, activation='relu')(x)
    out = Dense(X_normal.shape[1], activation='sigmoid')(x)
    ae  = Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(X_scaled, X_scaled, epochs=20, batch_size=8, verbose=0)
    autoencoders[(crop, region)] = ae

    # Learned Sensor Fusion
    seg_fusion = {}
    for (xf, yf) in [('rainfall_mm', 'soil_moisture_%'),
                      ('temperature_C', 'humidity_%'),
                      ('NDVI_index', 'soil_moisture_%')]:
        if len(normal_data) >= 5:
            xv = normal_data[xf].values.reshape(-1, 1)
            yv = normal_data[yf].values
            reg = LinearRegression().fit(xv, yv)
            residuals = yv - reg.predict(xv)
            seg_fusion[(xf, yf)] = {'model': reg, 'std': residuals.std() + 1e-6}
    fusion_models[(crop, region)] = seg_fusion

print("✅ Segment models trained")

# ============================================================
# 6. SCORE FUNCTIONS & NORMALIZATION
# ============================================================
def get_if_score(model, X):
    return -model.decision_function(X)[0]

def get_lof_score(model, X):
    return -model.decision_function(X)[0]

def get_ae_score(ae, scaler, X):
    Xs = scaler.transform(X)
    recon = ae.predict(Xs, verbose=0)
    return float(np.mean((Xs - recon) ** 2))

def get_ae_feature_scores(ae, scaler, X):
    Xs = scaler.transform(X)
    recon = ae.predict(Xs, verbose=0)
    return dict(zip(FEATURES, (Xs - recon)[0] ** 2))

def get_fusion_score(row_dict, key):
    if key not in fusion_models:
        return 0.0, {}
    seg = fusion_models[key]
    total_z, detail = 0.0, {}
    for (xf, yf), info in seg.items():
        expected = info['model'].predict([[row_dict[xf]]])[0]
        residual = abs(row_dict[yf] - expected)
        z = residual / info['std']
        total_z += z
        detail[f"{xf} → {yf}"] = round(z, 3)
    return total_z / max(len(seg), 1), detail

# Global normalization
all_if, all_lof, all_ae, all_fus = [], [], [], []
for _, row in df_eval.iterrows():
    key = (row['crop_type'], row['region'])
    if key not in if_models:
        continue
    X = np.array([row[FEATURES]])
    all_if.append(get_if_score(if_models[key], X))
    all_lof.append(get_lof_score(lof_models[key], X))
    all_ae.append(get_ae_score(autoencoders[key], scalers[key], X))
    fs, _ = get_fusion_score(row.to_dict(), key)
    all_fus.append(fs)

norm_if   = MinMaxScaler().fit(np.array(all_if).reshape(-1, 1))
norm_lof  = MinMaxScaler().fit(np.array(all_lof).reshape(-1, 1))
norm_ae   = MinMaxScaler().fit(np.array(all_ae).reshape(-1, 1))
norm_fus  = MinMaxScaler().fit(np.array(all_fus).reshape(-1, 1))
ae_thresh_raw = np.percentile(all_ae, 95)

# ============================================================
# 7. COMPUTE WEIGHTS FROM F1
# ============================================================
y_true_list, p_if, p_lof, p_ae, p_fus = [], [], [], [], []
for _, row in df_eval.iterrows():
    key = (row['crop_type'], row['region'])
    if key not in if_models:
        continue
    X = np.array([row[FEATURES]])
    y_true_list.append(row['is_anomaly'])
    p_if.append(1 if if_models[key].predict(X)[0] == -1 else 0)
    p_lof.append(1 if lof_models[key].predict(X)[0] == -1 else 0)
    p_ae.append(1 if get_ae_score(autoencoders[key], scalers[key], X) > ae_thresh_raw else 0)
    fs, _ = get_fusion_score(row.to_dict(), key)
    p_fus.append(1 if fs > 2.0 else 0)

y_true_arr = np.array(y_true_list)
f1s = {
    'if':     f1_score(y_true_arr, p_if,  zero_division=0),
    'lof':    f1_score(y_true_arr, p_lof, zero_division=0),
    'ae':     f1_score(y_true_arr, p_ae,  zero_division=0),
    'fusion': f1_score(y_true_arr, p_fus, zero_division=0),
}
total_f1 = sum(f1s.values()) + 1e-9
weights  = {k: v / total_f1 for k, v in f1s.items()}

# ============================================================
# 8. FIND OPTIMAL THRESHOLD
# ============================================================
all_scores = []
for _, row in df_eval.iterrows():
    key = (row['crop_type'], row['region'])
    if key not in if_models:
        continue
    X = np.array([row[FEATURES]])
    ifs  = norm_if.transform([[get_if_score(if_models[key], X)]])[0][0]
    lofs = norm_lof.transform([[get_lof_score(lof_models[key], X)]])[0][0]
    aes  = norm_ae.transform([[get_ae_score(autoencoders[key], scalers[key], X)]])[0][0]
    fs, _ = get_fusion_score(row.to_dict(), key)
    fuss = norm_fus.transform([[fs]])[0][0]
    score = (weights['if'] * ifs + weights['lof'] * lofs +
             weights['ae'] * aes + weights['fusion'] * fuss)
    all_scores.append(score)

df_eval_valid = df_eval[df_eval.apply(
    lambda r: (r['crop_type'], r['region']) in if_models, axis=1)].copy()
df_eval_valid['final_score'] = all_scores

best_thresh, best_f1_val = 0, 0
for t in np.linspace(0, 1, 200):
    pred = (df_eval_valid['final_score'] > t).astype(int)
    f1   = f1_score(df_eval_valid['is_anomaly'], pred, zero_division=0)
    if f1 > best_f1_val:
        best_f1_val = f1
        best_thresh = t

print(f"Optimal threshold: {best_thresh:.3f}  |  Best F1: {best_f1_val:.3f}")

# ============================================================
# 9. PERFORMANCE METRICS (for dashboard display)
# ============================================================
final_preds = (df_eval_valid['final_score'] > best_thresh).astype(int)
performance_metrics = {
    'f1':        round(f1_score(df_eval_valid['is_anomaly'], final_preds, zero_division=0), 3),
    'precision': round(precision_score(df_eval_valid['is_anomaly'], final_preds, zero_division=0), 3),
    'recall':    round(recall_score(df_eval_valid['is_anomaly'], final_preds, zero_division=0), 3),
    'auc_roc':   round(roc_auc_score(df_eval_valid['is_anomaly'], df_eval_valid['final_score']), 3),
}
print("Performance:", performance_metrics)

# ============================================================
# 10. YIELD MODEL
# ============================================================
df_yield = df_eval_valid[df_eval_valid['is_anomaly'] == 0].copy()
df_yield = df_yield.dropna(subset=YIELD_FEATURES + ['yield_kg_per_hectare'])
X_yield  = df_yield[YIELD_FEATURES].values
y_yield  = df_yield['yield_kg_per_hectare'].values

if XGB_AVAILABLE:
    yield_model = XGBRegressor(n_estimators=200, max_depth=4,
                               learning_rate=0.05, random_state=42, verbosity=0)
else:
    yield_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, random_state=42)
yield_model.fit(X_yield, y_yield)
print("✅ Yield model trained")

# ============================================================
# 11. SAVE EVERYTHING
# ============================================================
bundle = {
    'if_models':          if_models,
    'lof_models':         lof_models,
    'scalers':            scalers,
    'fusion_models':      fusion_models,
    'norm_if':            norm_if,
    'norm_lof':           norm_lof,
    'norm_ae':            norm_ae,
    'norm_fus':           norm_fus,
    'weights':            weights,
    'best_thresh':        best_thresh,
    'ae_thresh_raw':      ae_thresh_raw,
    'encoders':           encoders,
    'segment_stats':      segment_stats,
    'features':           FEATURES,
    'yield_features':     YIELD_FEATURES,
    'performance_metrics':performance_metrics,
    'yield_model':        yield_model,
}

with open("models/model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Save autoencoders separately (Keras models)
for (crop, region), ae in autoencoders.items():
    ae.save(f"models/ae_{crop}_{region}.keras")

print("✅ All models saved to models/")
print("\nFiles to upload to GitHub:")
print("  models/model_bundle.pkl")
for (crop, region) in autoencoders:
    print(f"  models/ae_{crop}_{region}.keras")
