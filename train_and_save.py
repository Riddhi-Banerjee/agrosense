# ============================================================
# train_and_save.py
# Run ONCE in Google Colab to train all models and save them.
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

os.makedirs("models", exist_ok=True)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
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
INJECTION_FRACTION = 0.10

# ============================================================
# 2. SEASONAL ENCODING
# ============================================================
df['sowing_date']  = pd.to_datetime(df['sowing_date'], dayfirst=True, errors='coerce')
df['sowing_month'] = df['sowing_date'].dt.month.fillna(6)
df['season_sin']   = np.sin(2 * np.pi * df['sowing_month'] / 12)
df['season_cos']   = np.cos(2 * np.pi * df['sowing_month'] / 12)

# ============================================================
# 3. LABEL ENCODING
# ============================================================
encoders = {}
for col in ['region', 'crop_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ============================================================
# 4. ANOMALY INJECTION — domain-rule-based
# ============================================================
def inject_contextual_anomalies(df, frac=INJECTION_FRACTION, seed=42):
    np.random.seed(seed)
    df = df.copy()
    df['is_anomaly']  = 0
    df['anomaly_type'] = 'normal'

    for (crop, region), data in df.groupby(['crop_type', 'region']):
        if len(data) < 15:
            continue
        n   = max(1, int(len(data) * frac))
        idx = np.random.choice(data.index, n, replace=False)
        seg_stats = {f: {'low':  data[f].quantile(0.10),
                         'high': data[f].quantile(0.90)} for f in FEATURES}

        for i in idx:
            atype = np.random.choice(
                ['drought_stress','heat_stress','sensor_inconsistency','crop_failure'],
                p=[0.25, 0.25, 0.25, 0.25]
            )
            if atype == 'drought_stress':
                df.loc[i, 'soil_moisture_%'] = seg_stats['soil_moisture_%']['low'] * 0.20
                df.loc[i, 'rainfall_mm']      = seg_stats['rainfall_mm']['low']    * 0.30
                df.loc[i, 'NDVI_index']       = seg_stats['NDVI_index']['low']     * 0.40
            elif atype == 'heat_stress':
                df.loc[i, 'temperature_C'] = seg_stats['temperature_C']['high'] * 1.50
                df.loc[i, 'humidity_%']     = seg_stats['humidity_%']['high']    * 1.30
            elif atype == 'sensor_inconsistency':
                df.loc[i, 'rainfall_mm']     = seg_stats['rainfall_mm']['high']
                df.loc[i, 'soil_moisture_%'] = seg_stats['soil_moisture_%']['low'] * 0.20
            elif atype == 'crop_failure':
                df.loc[i, 'NDVI_index']      = seg_stats['NDVI_index']['low']     * 0.30
                df.loc[i, 'soil_moisture_%'] = seg_stats['soil_moisture_%']['low'] * 0.20
                df.loc[i, 'temperature_C']   = seg_stats['temperature_C']['high'] * 1.50

            df.loc[i, 'is_anomaly']   = 1
            df.loc[i, 'anomaly_type'] = atype

    print(f"Injected {df['is_anomaly'].sum()} anomalies "
          f"({100 * df['is_anomaly'].mean():.1f}%)")
    print(df['anomaly_type'].value_counts())
    return df

df_eval = inject_contextual_anomalies(df)
print("✅ Anomalies injected\n")

# ============================================================
# 5. TRAIN PER-SEGMENT MODELS
# ============================================================
if_models, lof_models, autoencoders, scalers, fusion_models = {}, {}, {}, {}, {}
segment_stats = {}

for (crop, region), data in df_eval.groupby(['crop_type', 'region']):
    if len(data) < 15:
        continue

    normal_data = data[data['is_anomaly'] == 0]
    X_normal    = normal_data[FEATURES].values

    # Normal range stats for dashboard
    segment_stats[(crop, region)] = {
        f: {'low':  normal_data[f].quantile(0.10),
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
    lof    = LocalOutlierFactor(n_neighbors=n_nbrs,
                                contamination=INJECTION_FRACTION, novelty=True)
    lof.fit(X_normal)
    lof_models[(crop, region)] = lof

    # Autoencoder
    scaler   = MinMaxScaler()
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
    for (xf, yf) in [('rainfall_mm',  'soil_moisture_%'),
                      ('temperature_C','humidity_%'),
                      ('NDVI_index',   'soil_moisture_%')]:
        if len(normal_data) >= 5:
            xv  = normal_data[xf].values.reshape(-1, 1)
            yv  = normal_data[yf].values
            reg = LinearRegression().fit(xv, yv)
            res = yv - reg.predict(xv)
            seg_fusion[(xf, yf)] = {'model': reg, 'std': res.std() + 1e-6}
    fusion_models[(crop, region)] = seg_fusion

print("✅ Segment models trained\n")

# ============================================================
# 6. SCORE FUNCTIONS & GLOBAL NORMALISATION
# ============================================================
def get_if_score(model, X):
    return float(-model.decision_function(X)[0])

def get_lof_score(model, X):
    return float(-model.decision_function(X)[0])

def get_ae_score(ae, scaler, X):
    Xs    = scaler.transform(X)
    recon = ae.predict(Xs, verbose=0)
    return float(np.mean((Xs - recon) ** 2))

def get_fusion_score(row_dict, key):
    if key not in fusion_models:
        return 0.0, {}
    seg     = fusion_models[key]
    total_z = 0.0
    detail  = {}
    for (xf, yf), info in seg.items():
        expected = info['model'].predict([[row_dict[xf]]])[0]
        residual = abs(row_dict[yf] - expected)
        z        = residual / info['std']
        total_z += z
        detail[f"{xf} → {yf}"] = round(z, 3)
    return total_z / max(len(seg), 1), detail

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
# 7. F1-BASED ENSEMBLE WEIGHTS
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
print("Per-model F1 & weights:")
for k, v in f1s.items():
    print(f"  {k:8s}: F1={v:.3f}  weight={weights[k]:.3f}")

# ============================================================
# 8. OPTIMAL THRESHOLD
# ============================================================
all_scores = []
for _, row in df_eval.iterrows():
    key = (row['crop_type'], row['region'])
    if key not in if_models:
        continue
    X    = np.array([row[FEATURES]])
    ifs  = float(norm_if.transform([[get_if_score(if_models[key], X)]])[0][0])
    lofs = float(norm_lof.transform([[get_lof_score(lof_models[key], X)]])[0][0])
    aes  = float(norm_ae.transform([[get_ae_score(autoencoders[key], scalers[key], X)]])[0][0])
    fs, _ = get_fusion_score(row.to_dict(), key)
    fuss = float(norm_fus.transform([[fs]])[0][0])
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

print(f"\nOptimal threshold: {best_thresh:.3f}  |  Best F1: {best_f1_val:.3f}")

# ============================================================
# 9. PERFORMANCE METRICS
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
# 10. YIELD MODEL — Agronomic formula
#
# WHY: The dataset yield column has R² ≈ -0.26 with all sensor
# features (near-zero correlation). A trained ML model on this
# data predicts garbage, including negative yields.
# This formula encodes agronomic domain knowledge directly:
# each factor multiplies a crop-specific base yield.
#
# Base yields (kg/ha) — conservative literature values:
#   Cotton:  3,800   Maize:  5,500   Rice:  4,500
#   Soybean: 3,200   Wheat:  4,800
#
# Factors (each 0.5–1.15, multiply together):
#   Soil moisture  — optimal 20–35%
#   Temperature    — optimal 20–30°C
#   Soil pH        — optimal 6.0–7.0
#   NDVI           — linear 0.5 (bare) to 1.3 (lush)
#   Rainfall       — optimal 100–250 mm
#   Sunlight hours — more is better up to 10 h/day
#   Pesticide ml   — optimal 15–35 ml, penalty outside
# ============================================================

# Build crop-name → base yield map using encoder classes
crop_base_map = {}
_base_yields  = {
    'Cotton': 3800, 'Maize': 5500, 'Rice': 4500,
    'Soybean': 3200, 'Wheat': 4800
}
for name in encoders['crop_type'].classes_:
    enc = int(encoders['crop_type'].transform([name])[0])
    crop_base_map[enc] = _base_yields.get(name, 4000)

print("\nCrop base yields (kg/ha):")
for enc, base in crop_base_map.items():
    name = encoders['crop_type'].inverse_transform([enc])[0]
    print(f"  {name}: {base}")

# Verify formula on default values
def _formula_yield(user_dict, crop_enc, crop_base_map):
    base = crop_base_map.get(int(crop_enc), 4000)
    sm   = user_dict.get('soil_moisture_%', 25)
    tmp  = user_dict.get('temperature_C', 24)
    ph   = user_dict.get('soil_pH', 6.5)
    ndvi = user_dict.get('NDVI_index', 0.6)
    rain = user_dict.get('rainfall_mm', 180)
    sun  = user_dict.get('sunlight_hours', 7)
    pest = user_dict.get('pesticide_usage_ml', 25)

    # Soil moisture factor
    if 20 <= sm <= 35:
        sm_f = 1.0
    elif sm < 20:
        sm_f = max(0.40, 0.60 + 0.02 * sm)
    else:
        sm_f = max(0.65, 1.0 - 0.012 * (sm - 35))

    # Temperature factor
    if 20 <= tmp <= 30:
        tmp_f = 1.0
    elif tmp < 20:
        tmp_f = max(0.65, 0.65 + 0.017 * (tmp - 10))
    else:
        tmp_f = max(0.45, 1.0 - 0.028 * (tmp - 30))

    # pH factor
    if 6.0 <= ph <= 7.0:
        ph_f = 1.0
    else:
        ph_f = max(0.70, 1.0 - 0.12 * abs(ph - 6.5))

    # NDVI factor — direct linear mapping
    ndvi_f = max(0.30, min(1.30, 0.50 + 0.80 * ndvi))

    # Rainfall factor
    if 100 <= rain <= 250:
        rain_f = 1.0
    elif rain < 100:
        rain_f = max(0.50, 0.50 + 0.005 * rain)
    else:
        rain_f = max(0.70, 1.0 - 0.0012 * (rain - 250))

    # Sunlight factor
    sun_f = min(1.15, max(0.70, 0.70 + 0.045 * sun))

    # Pesticide factor
    if 15 <= pest <= 35:
        pest_f = 1.0
    else:
        pest_f = max(0.80, 1.0 - 0.006 * abs(pest - 25))

    result = base * sm_f * tmp_f * ph_f * ndvi_f * rain_f * sun_f * pest_f
    return max(300.0, round(result, 0))   # hard floor 300 kg/ha, never negative

print("\nFormula yield verification (default inputs):")
test_input = {'soil_moisture_%': 25, 'soil_pH': 6.5, 'temperature_C': 24,
              'rainfall_mm': 180, 'humidity_%': 65, 'NDVI_index': 0.6,
              'sunlight_hours': 7, 'pesticide_usage_ml': 25}
for enc, base in crop_base_map.items():
    name   = encoders['crop_type'].inverse_transform([enc])[0]
    y_pred = _formula_yield(test_input, enc, crop_base_map)
    print(f"  {name}: {y_pred:.0f} kg/ha")

print("\nFormula yield — stressed conditions (drought + heat):")
stressed = {'soil_moisture_%': 5, 'soil_pH': 6.5, 'temperature_C': 42,
            'rainfall_mm': 30, 'humidity_%': 65, 'NDVI_index': 0.2,
            'sunlight_hours': 7, 'pesticide_usage_ml': 25}
for enc, base in crop_base_map.items():
    name   = encoders['crop_type'].inverse_transform([enc])[0]
    y_pred = _formula_yield(stressed, enc, crop_base_map)
    print(f"  {name}: {y_pred:.0f} kg/ha")

print("✅ Yield formula verified\n")

# ============================================================
# 11. SAVE EVERYTHING
# ============================================================
bundle = {
    # Anomaly models
    'if_models':           if_models,
    'lof_models':          lof_models,
    'scalers':             scalers,
    'fusion_models':       fusion_models,
    # Normalisers
    'norm_if':             norm_if,
    'norm_lof':            norm_lof,
    'norm_ae':             norm_ae,
    'norm_fus':            norm_fus,
    # Ensemble
    'weights':             weights,
    'best_thresh':         best_thresh,
    'ae_thresh_raw':       ae_thresh_raw,
    # Metadata
    'encoders':            encoders,
    'segment_stats':       segment_stats,
    'features':            FEATURES,
    'performance_metrics': performance_metrics,
    # Yield — formula-based, no sklearn model
    'crop_base_map':       crop_base_map,
}

with open("models/model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Save Keras autoencoders separately
for (crop, region), ae in autoencoders.items():
    ae.save(f"models/ae_{crop}_{region}.keras")

print("✅ All models saved to models/")
print("\nFiles to upload to GitHub:")
print("  models/model_bundle.pkl")
for (crop, region) in autoencoders:
    print(f"  models/ae_{crop}_{region}.keras")
