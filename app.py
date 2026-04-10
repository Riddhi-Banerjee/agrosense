import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    load_model = None
    TF_AVAILABLE = False
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AgroSense — Smart Farm Anomaly Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.main { background-color: #0d1117; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #0f1e12 50%, #0d1117 100%); }

/* FIX 1 — sidebar text white */
section[data-testid="stSidebar"] {
    background: #0f1a12;
    border-right: 1px solid #1e3a28;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #e8f5e3 !important;
}
section[data-testid="stSidebar"] h3 { color: #4caf72 !important; }
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: #9ab5a0 !important;
}
.stSlider > div > div > div > div { background: #4caf72 !important; }

/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800; color: #e8f5e3;
    letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #4caf72;
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #161d1a; border: 1px solid #1e3a28;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #4caf72; }
.metric-card .lbl {
    font-size: 0.72rem; color: #6b8f73;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;
}

/* Result banners */
.result-anomaly {
    background: linear-gradient(135deg, #2d0f0f, #1a0808);
    border: 1.5px solid #e53935; border-radius: 14px;
    padding: 1.5rem 2rem; margin: 1rem 0;
}
.result-normal {
    background: linear-gradient(135deg, #0a2615, #061a0e);
    border: 1.5px solid #4caf72; border-radius: 14px;
    padding: 1.5rem 2rem; margin: 1rem 0;
}
.result-borderline {
    background: linear-gradient(135deg, #2d2200, #1a1500);
    border: 1.5px solid #f5a623; border-radius: 14px;
    padding: 1.5rem 2rem; margin: 1rem 0;
}
.result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-score {
    font-family: 'DM Mono', monospace; font-size: 0.85rem;
    color: #9ab5a0; margin-top: 0.3rem;
}

/* Section header */
.section-head {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: #4caf72;
    border-bottom: 1px solid #1e3a28; padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sensor analysis cards */
.sensor-card {
    background: #161d1a; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
    border-left: 4px solid #1e3a28;
}
.sensor-card.anomaly { border-left-color: #ef5350; }
.sensor-card.warning { border-left-color: #f5a623; }
.sensor-card.ok      { border-left-color: #4caf72; }
.sc-feature {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #6b8f73; margin-bottom: 4px;
}
.sc-value { font-size: 1.3rem; font-weight: 700; color: #e8f5e3; margin-bottom: 2px; }
.sc-status { font-size: 0.82rem; margin-bottom: 4px; }
.sc-desc { font-size: 0.78rem; color: #9ab5a0; line-height: 1.55; }

/* Recommendation cards */
.rec-card {
    background: #0f1e12; border: 1px solid #1e3a28;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.rec-card.urgent  { border-color: #ef5350; background: #1a0808; }
.rec-card.caution { border-color: #f5a623; background: #1a1200; }
.rec-card.info    { border-color: #378ADD; background: #0a1520; }
.rc-num  { font-family:'DM Mono',monospace; font-size:0.7rem; color:#4caf72; margin-bottom:4px; }
.rc-title { font-size:0.95rem; font-weight:600; color:#e8f5e3; margin-bottom:4px; }
.rc-desc  { font-size:0.8rem; color:#9ab5a0; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_bundle():
    with open("models/model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    autoencoders = {}
    for fname in os.listdir("models"):
        if fname.startswith("ae_") and fname.endswith(".keras"):
            parts  = fname.replace(".keras", "").split("_")
            crop   = int(parts[1])
            region = int(parts[2])
            if TF_AVAILABLE:
                autoencoders[(crop, region)] = load_model(f"models/{fname}", compile=False)
            else:
                autoencoders[(crop, region)] = None
    bundle['autoencoders'] = autoencoders
    return bundle

bundle        = load_bundle()
if_models     = bundle['if_models']
lof_models    = bundle['lof_models']
autoencoders  = bundle['autoencoders']
scalers       = bundle['scalers']
fusion_models = bundle['fusion_models']
norm_if       = bundle['norm_if']
norm_lof      = bundle['norm_lof']
norm_ae       = bundle['norm_ae']
norm_fus      = bundle['norm_fus']
weights       = bundle['weights']
best_thresh   = bundle['best_thresh']
ae_thresh_raw = bundle['ae_thresh_raw']
encoders      = bundle['encoders']
segment_stats = bundle['segment_stats']
FEATURES      = bundle['features']
YIELD_FEATURES= bundle['yield_features']
perf          = bundle['performance_metrics']
yield_model   = bundle['yield_model']

CROP_LIST   = list(encoders['crop_type'].classes_)
REGION_LIST = list(encoders['region'].classes_)

# ============================================================
# HELPERS — SCORES
# ============================================================
def encode_val(val, encoder):
    if val in encoder.classes_:
        return int(encoder.transform([val])[0])
    return None

def get_if_score(model, X):
    return float(-model.decision_function(X)[0])

def get_lof_score(model, X):
    return float(-model.decision_function(X)[0])

def get_ae_score(ae, scaler, X):
    if ae is None: return 0.0
    Xs = scaler.transform(X)
    recon = ae.predict(Xs, verbose=0)
    return float(np.mean((Xs - recon) ** 2))

def get_ae_feature_scores(ae, scaler, X):
    if ae is None: return {f: 0.0 for f in FEATURES}
    Xs = scaler.transform(X)
    recon = ae.predict(Xs, verbose=0)
    return dict(zip(FEATURES, (Xs - recon)[0] ** 2))

def get_fusion_score(row_dict, key):
    if key not in fusion_models: return 0.0, {}
    seg = fusion_models[key]
    total_z, detail = 0.0, {}
    for (xf, yf), info in seg.items():
        expected = info['model'].predict([[row_dict[xf]]])[0]
        residual = abs(row_dict[yf] - expected)
        z = residual / info['std']
        total_z += z
        detail[f"{xf} → {yf}"] = round(z, 3)
    return total_z / max(len(seg), 1), detail

def predict(user_data):
    crop_enc   = encode_val(user_data['crop_type'], encoders['crop_type'])
    region_enc = encode_val(user_data['region'],    encoders['region'])
    if crop_enc is None or region_enc is None: return None
    key = (crop_enc, region_enc)
    if key not in if_models: return None

    X = np.array([[user_data[f] for f in FEATURES]])

    if_raw              = get_if_score(if_models[key], X)
    lof_raw             = get_lof_score(lof_models[key], X)
    ae_raw              = get_ae_score(autoencoders[key], scalers[key], X)
    fs_raw, fus_detail  = get_fusion_score(user_data, key)

    if_s  = float(norm_if.transform([[if_raw]])[0][0])
    lof_s = float(norm_lof.transform([[lof_raw]])[0][0])
    ae_s  = float(norm_ae.transform([[ae_raw]])[0][0])
    fus_s = float(norm_fus.transform([[fs_raw]])[0][0])

    final_score = (weights['if']     * if_s  +
                   weights['lof']    * lof_s +
                   weights['ae']     * ae_s  +
                   weights['fusion'] * fus_s)

    seg_s        = segment_stats.get(key, {})
    param_issues = []
    for f in FEATURES:
        if f not in seg_s: continue
        val  = user_data[f]
        low  = seg_s[f]['low']
        high = seg_s[f]['high']
        mean = seg_s[f]['mean']
        if val < low:
            param_issues.append({'feature':f,'value':val,'status':'LOW',
                                  'low':low,'high':high,'mean':mean})
        elif val > high:
            param_issues.append({'feature':f,'value':val,'status':'HIGH',
                                  'low':low,'high':high,'mean':mean})

    extra = {
        'sunlight_hours':     user_data.get('sunlight_hours', 7.0),
        'pesticide_usage_ml': user_data.get('pesticide_usage_ml', 25.0),
        'total_days':         user_data.get('total_days', 120),
        'season_sin':         user_data.get('season_sin', 0.0),
        'season_cos':         user_data.get('season_cos', 1.0),
        'crop_type':          crop_enc,
        'region':             region_enc,
    }
    yield_input     = np.array([[user_data[f] for f in FEATURES] +
                                 [extra[k] for k in ['sunlight_hours','pesticide_usage_ml',
                                                      'total_days','season_sin','season_cos',
                                                      'crop_type','region']]])
    predicted_yield = float(yield_model.predict(yield_input)[0])

    return {
        'final_score': final_score,
        'if_s': if_s, 'lof_s': lof_s, 'ae_s': ae_s, 'fus_s': fus_s,
        'fusion_detail':  fus_detail,
        'ae_feat':        get_ae_feature_scores(autoencoders[key], scalers[key], X),
        'param_issues':   param_issues,
        'predicted_yield':predicted_yield,
        'seg_stats':      seg_s,
        'key':            key,
    }

# ============================================================
# HELPERS — PLAIN ENGLISH
# ============================================================
FEATURE_LABELS = {
    'soil_moisture_%': 'Soil Moisture',
    'soil_pH':         'Soil pH',
    'temperature_C':   'Temperature',
    'rainfall_mm':     'Rainfall',
    'humidity_%':      'Humidity',
    'NDVI_index':      'NDVI (Crop Health Index)',
}
FEATURE_UNITS = {
    'soil_moisture_%': '%',
    'soil_pH':         '',
    'temperature_C':   '°C',
    'rainfall_mm':     ' mm',
    'humidity_%':      '%',
    'NDVI_index':      '',
}

def feature_plain_english(f, val, status, low, high, crop, region):
    unit = FEATURE_UNITS.get(f, '')
    expl = {
        'soil_moisture_%': {
            'LOW':  (f"Soil moisture is critically low at {val:.1f}%. The healthy range for "
                     f"{crop} in {region} is {low:.1f}–{high:.1f}%. Dry soil causes root stress, "
                     f"wilting and reduced nutrient uptake, which directly lowers yield."),
            'HIGH': (f"Soil moisture is too high at {val:.1f}% (normal: {low:.1f}–{high:.1f}%). "
                     f"Excess water drowns roots, promotes fungal disease such as root rot, "
                     f"and can make the field unworkable."),
        },
        'soil_pH': {
            'LOW':  (f"Soil pH is {val:.2f} — too acidic for {crop} (normal: {low:.2f}–{high:.2f}). "
                     f"Acidic soil blocks phosphorus and calcium absorption, leading to "
                     f"stunted growth and poor grain quality."),
            'HIGH': (f"Soil pH is {val:.2f} — too alkaline (normal: {low:.2f}–{high:.2f}). "
                     f"Alkaline conditions lock out iron and manganese, causing yellowing "
                     f"leaves (chlorosis) and slow development in {crop}."),
        },
        'temperature_C': {
            'LOW':  (f"Temperature is {val:.1f}°C — below the safe threshold for {crop} "
                     f"in {region} (normal: {low:.1f}–{high:.1f}°C). Cold slows photosynthesis, "
                     f"delays flowering, and can damage sensitive growing tissue overnight."),
            'HIGH': (f"Temperature is {val:.1f}°C — critically above the safe range of "
                     f"{low:.1f}–{high:.1f}°C for {crop}. Extreme heat damages pollen viability, "
                     f"accelerates moisture loss and can cause irreversible yield loss if "
                     f"sustained for more than 2–3 days."),
        },
        'rainfall_mm': {
            'LOW':  (f"Rainfall is only {val:.0f} mm against the expected {low:.0f}–{high:.0f} mm "
                     f"for {crop} in {region}. This moisture deficit will stress the crop and "
                     f"requires immediate supplemental irrigation to prevent yield loss."),
            'HIGH': (f"Rainfall of {val:.0f} mm greatly exceeds the {low:.0f}–{high:.0f} mm "
                     f"normal range. Flooding risk is high — excess water leaches nutrients, "
                     f"compacts soil structure, and triggers fungal and bacterial disease."),
        },
        'humidity_%': {
            'LOW':  (f"Humidity is {val:.1f}% — lower than the {low:.1f}–{high:.1f}% range "
                     f"expected for {crop}. Low humidity increases leaf transpiration and "
                     f"water stress, particularly during the hottest part of the day."),
            'HIGH': (f"Humidity is {val:.1f}% — above the safe {low:.1f}–{high:.1f}% range. "
                     f"High humidity combined with warm temperatures creates ideal conditions "
                     f"for blight, mildew, and other fungal diseases in {crop}."),
        },
        'NDVI_index': {
            'LOW':  (f"NDVI is {val:.2f} — well below the healthy {low:.2f}–{high:.2f} range. "
                     f"A low NDVI signals poor canopy coverage. This typically means the crop "
                     f"is stressed, diseased, pest-damaged, or has very sparse plant coverage."),
            'HIGH': (f"NDVI is {val:.2f} — above the expected {low:.2f}–{high:.2f}. "
                     f"While generally a sign of lush growth, an unusually high reading can "
                     f"indicate sensor miscalibration or weed overgrowth suppressing {crop}."),
        },
    }
    default = (f"{FEATURE_LABELS.get(f,f)} reading of {val:.2f}{unit} is outside the "
               f"expected range of {low:.2f}–{high:.2f}{unit} for {crop} in {region}.")
    return expl.get(f, {}).get(status, default)


def fusion_plain_english(pair, z, crop, region):
    sev = "critically" if z > 3.0 else "significantly" if z > 2.0 else "slightly"
    expl = {
        'rainfall_mm → soil_moisture_%': (
            f"Given the rainfall reading, the expected soil moisture is {sev} different from "
            f"what the soil sensor reports (deviation: {z:.1f}σ). For {crop} in {region}, "
            f"this level of rain should produce higher soil moisture. Likely causes: "
            f"sensor fault, rapid drainage, or an error in the rainfall data."),
        'temperature_C → humidity_%': (
            f"The temperature and humidity readings are {sev} inconsistent (deviation: {z:.1f}σ). "
            f"At this temperature, atmospheric humidity should be within a predictable range. "
            f"A faulty humidity sensor or an unusual microclimate event may explain this gap."),
        'NDVI_index → soil_moisture_%': (
            f"Crop health (NDVI) and soil moisture are {sev} inconsistent (deviation: {z:.1f}σ). "
            f"A crop showing this level of NDVI typically requires more soil moisture than "
            f"currently reported. This may indicate a soil sensor fault or highly efficient "
            f"water use by an unusually deep-rooted crop."),
    }
    return expl.get(pair,
        f"Sensor pair '{pair}' shows a {sev} inconsistency (z-score: {z:.1f}σ). "
        f"Cross-check both sensors manually.")


def generate_recommendations(param_issues, fusion_detail, score,
                              crop, region, predicted_yield, best_thresh):
    recs = []

    if score > best_thresh:
        recs.append({
            'type': 'urgent',
            'title': 'Immediate field inspection required',
            'desc': (f"The anomaly score ({score:.3f}) exceeds the detection threshold "
                     f"({best_thresh:.3f}). Multiple sensors are reporting abnormal conditions. "
                     f"Visit the field within 24 hours to physically verify readings and "
                     f"check visible crop health before taking corrective action.")
        })

    feature_recs = {
        ('soil_moisture_%','LOW'):  ('urgent',  'Activate emergency irrigation now',
            f"Soil moisture is critically low. Start drip or sprinkler irrigation within 6–12 hours. "
            f"Target the 20–35% range and monitor every 12 hours until stable."),
        ('soil_moisture_%','HIGH'): ('caution', 'Improve field drainage immediately',
            f"Pause all irrigation. Clear drainage channels and create furrows to redirect excess water. "
            f"Watch for early root rot symptoms such as yellowing lower leaves."),
        ('temperature_C','HIGH'):   ('urgent',  'Apply heat stress mitigation',
            f"Apply reflective mulch, increase irrigation frequency, and avoid field work during "
            f"peak heat hours. Consider temporary shade netting for young plants if available."),
        ('temperature_C','LOW'):    ('caution', 'Protect crops from cold stress',
            f"Use frost protection cloth or row covers overnight. Delay transplanting or sowing "
            f"until temperatures return to the normal range."),
        ('soil_pH','LOW'):          ('caution', 'Apply lime to correct soil acidity',
            f"Apply agricultural lime at 1–2 tonnes per hectare. Retest pH after 4–6 weeks. "
            f"Avoid ammonium-based fertilisers, which further acidify the soil."),
        ('soil_pH','HIGH'):         ('caution', 'Apply sulphur to reduce soil alkalinity',
            f"Apply elemental sulphur or acidifying fertiliser such as ammonium sulphate. "
            f"Add organic compost to gradually lower pH over the growing season."),
        ('rainfall_mm','LOW'):      ('urgent',  'Switch to supplemental irrigation — drought risk',
            f"Rainfall is far below requirements for {crop} in {region}. Begin scheduled irrigation "
            f"immediately and apply mulch to reduce evaporation losses."),
        ('rainfall_mm','HIGH'):     ('caution', 'Monitor for flood damage and disease outbreak',
            f"Inspect and clear drainage systems. Apply preventive fungicide within 48 hours "
            f"as wet conditions significantly increase disease risk."),
        ('humidity_%','HIGH'):      ('caution', 'Apply preventive fungicide — disease risk elevated',
            f"High humidity creates conditions ideal for fungal disease. Spray a broad-spectrum "
            f"fungicide as a preventive measure and improve canopy air circulation if possible."),
        ('NDVI_index','LOW'):       ('urgent',  'Investigate crop health decline urgently',
            f"NDVI is critically low — inspect the field for disease, pest damage, or nutrient "
            f"deficiency. Collect leaf samples for laboratory analysis if no visible cause is found."),
    }

    for iss in param_issues:
        k = (iss['feature'], iss['status'])
        if k in feature_recs:
            t, title, desc = feature_recs[k]
            recs.append({'type': t, 'title': title, 'desc': desc})

    for pair, z in fusion_detail.items():
        if z > 2.0:
            sensor_name = pair.split('→')[0].strip().replace('_', ' ')
            recs.append({
                'type': 'info',
                'title': f'Verify sensor calibration — {sensor_name}',
                'desc': (f"The {pair} relationship is statistically inconsistent (z-score: {z:.1f}σ). "
                         f"This may indicate a faulty sensor rather than a real field condition. "
                         f"Take a manual measurement before acting on this alert.")
            })

    if predicted_yield < 3000:
        recs.append({
            'type': 'urgent',
            'title': 'Predicted yield critically low — address anomalies urgently',
            'desc': (f"Model predicts {predicted_yield:.0f} kg/ha, well below the average "
                     f"of ~4,033 kg/ha. Current conditions are expected to cause significant "
                     f"yield loss. Prioritise the urgent actions above.")
        })
    elif predicted_yield > 4500:
        recs.append({
            'type': 'info',
            'title': 'Yield outlook is positive — protect current conditions',
            'desc': (f"Predicted yield of {predicted_yield:.0f} kg/ha is above average. "
                     f"Maintain current management practices and address any flagged issues "
                     f"promptly to protect this yield potential.")
        })

    if not recs:
        recs.append({
            'type': 'info',
            'title': 'All conditions normal — continue routine monitoring',
            'desc': (f"No anomalies detected for {crop} in {region}. Continue regular monitoring "
                     f"at your standard inspection schedule. Recommended next check: 48–72 hours.")
        })

    return recs

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem;color:#e8f5e3;">🌾 AgroSense</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Farm Context")
    crop_sel   = st.selectbox("Crop Type", CROP_LIST)
    region_sel = st.selectbox("Region",    REGION_LIST)

    st.markdown("### Sensor Readings")
    soil_moist  = st.slider("Soil Moisture (%)",   0.0,  60.0, 25.0, 0.1)
    soil_ph     = st.slider("Soil pH",             4.0,   9.0,  6.5, 0.01)
    temperature = st.slider("Temperature (°C)",   10.0,  55.0, 24.0, 0.1)
    rainfall    = st.slider("Rainfall (mm)",        0.0, 400.0,180.0, 1.0)
    humidity    = st.slider("Humidity (%)",         0.0, 100.0, 65.0, 0.1)
    ndvi        = st.slider("NDVI Index",           0.0,   1.0,  0.6, 0.01)

    st.markdown("### Farm Details")
    sunlight   = st.slider("Sunlight Hours/day",  2.0,  14.0,  7.0, 0.1)
    pesticide  = st.slider("Pesticide (ml)",       0.0,  60.0, 25.0, 0.5)
    total_days = st.slider("Growing Days",         60,   180,  120,   1)
    sow_month  = st.selectbox("Sowing Month",
                              ['Jan','Feb','Mar','Apr','May','Jun',
                               'Jul','Aug','Sep','Oct','Nov','Dec'])
    month_num  = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec'].index(sow_month) + 1

    run_btn = st.button("🔍 Analyse Farm Conditions", use_container_width=True)

# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown('<div class="hero-title">AgroSense Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Hybrid Contextual Anomaly Detection · Precision Agriculture</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, label, val in zip(
    [c1, c2, c3, c4],
    ['F1 Score', 'Precision', 'Recall', 'AUC-ROC'],
    [perf['f1'], perf['precision'], perf['recall'], perf['auc_roc']]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="val">{val}</div>
            <div class="lbl">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ============================================================
# RESULTS
# ============================================================
if run_btn:
    user_data = {
        'crop_type':          crop_sel,
        'region':             region_sel,
        'soil_moisture_%':    soil_moist,
        'soil_pH':            soil_ph,
        'temperature_C':      temperature,
        'rainfall_mm':        rainfall,
        'humidity_%':         humidity,
        'NDVI_index':         ndvi,
        'sunlight_hours':     sunlight,
        'pesticide_usage_ml': pesticide,
        'total_days':         total_days,
        'season_sin':         np.sin(2 * np.pi * month_num / 12),
        'season_cos':         np.cos(2 * np.pi * month_num / 12),
    }

    result = predict(user_data)

    if result is None:
        st.error("No trained model found for this crop-region combination.")
    else:
        score = result['final_score']

        # ---- Decision Banner ----
        if score > best_thresh:
            st.markdown(f"""
            <div class="result-anomaly">
                <div class="result-title" style="color:#ef5350;">⚠ Anomaly Detected</div>
                <div class="result-score">
                    Ensemble Score: {score:.3f} &nbsp;|&nbsp; Threshold: {best_thresh:.3f}
                </div>
                <div style="font-size:0.85rem;color:#ef9a9a;margin-top:0.5rem;">
                    Abnormal farming conditions identified for
                    <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>.
                    Review sensor analysis and follow recommended actions below.
                </div>
            </div>""", unsafe_allow_html=True)
        elif score >= 0.5:
            st.markdown(f"""
            <div class="result-borderline">
                <div class="result-title" style="color:#f5a623;">⚡ Borderline Conditions</div>
                <div class="result-score">
                    Ensemble Score: {score:.3f} &nbsp;|&nbsp; Threshold: {best_thresh:.3f}
                </div>
                <div style="font-size:0.85rem;color:#ffe082;margin-top:0.5rem;">
                    Readings are approaching the anomalous range for
                    <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>.
                    Monitor closely and review recommendations below.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                <div class="result-title" style="color:#66bb6a;">✓ Conditions Normal</div>
                <div class="result-score">
                    Ensemble Score: {score:.3f} &nbsp;|&nbsp; Threshold: {best_thresh:.3f}
                </div>
                <div style="font-size:0.85rem;color:#a5d6a7;margin-top:0.5rem;">
                    All sensor readings are within the expected range for
                    <strong>{crop_sel}</strong> in <strong>{region_sel}</strong>.
                </div>
            </div>""", unsafe_allow_html=True)

        # ============================================================
        # TWO COLUMN LAYOUT
        # ============================================================
        left_col, right_col = st.columns([1, 1])

        # ---- LEFT: Sensor-by-Sensor Analysis + Fusion ----
        with left_col:
            st.markdown('<div class="section-head">Sensor-by-Sensor Analysis</div>',
                        unsafe_allow_html=True)

            seg_s          = result['seg_stats']
            issues_map     = {iss['feature']: iss for iss in result['param_issues']}

            for f in FEATURES:
                if f not in seg_s: continue
                val   = user_data[f]
                low   = seg_s[f]['low']
                high  = seg_s[f]['high']
                unit  = FEATURE_UNITS.get(f, '')
                label = FEATURE_LABELS.get(f, f)

                if f in issues_map:
                    iss    = issues_map[f]
                    status = iss['status']
                    icon   = '🔴' if status == 'HIGH' else '🔵'
                    status_html = (f'<span style="color:#ef5350;font-weight:600;">'
                                   f'{icon} {status} — outside normal range '
                                   f'({low:.2f}–{high:.2f}{unit})</span>')
                    desc = feature_plain_english(f, val, status, low, high,
                                                 crop_sel, region_sel)
                    card_cls = 'anomaly'
                else:
                    status_html = (f'<span style="color:#4caf72;font-weight:600;">'
                                   f'✓ Normal — within range ({low:.2f}–{high:.2f}{unit})</span>')
                    desc = (f"Reading of {val:.2f}{unit} is within the expected range "
                            f"for {crop_sel} in {region_sel}. No action needed.")
                    card_cls = 'ok'

                st.markdown(f"""
                <div class="sensor-card {card_cls}">
                    <div class="sc-feature">{label}</div>
                    <div class="sc-value">{val:.2f}{unit}</div>
                    <div class="sc-status">{status_html}</div>
                    <div class="sc-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

            # Sensor Fusion
            st.markdown('<div class="section-head">Sensor Consistency Analysis</div>',
                        unsafe_allow_html=True)

            fd = result['fusion_detail']
            if fd:
                for pair, z in fd.items():
                    if z > 2.0:
                        card_cls   = 'anomaly'
                        z_html     = (f'<span style="color:#ef5350;font-weight:600;">'
                                      f'⚠ Inconsistent — {z:.1f}σ above expected</span>')
                    elif z > 1.0:
                        card_cls   = 'warning'
                        z_html     = (f'<span style="color:#f5a623;font-weight:600;">'
                                      f'⚡ Slight deviation — {z:.1f}σ</span>')
                    else:
                        card_cls   = 'ok'
                        z_html     = (f'<span style="color:#4caf72;font-weight:600;">'
                                      f'✓ Consistent — {z:.1f}σ (normal)</span>')

                    pair_label = pair.replace('_', ' ').replace('%', '')
                    desc = fusion_plain_english(pair, z, crop_sel, region_sel)

                    st.markdown(f"""
                    <div class="sensor-card {card_cls}">
                        <div class="sc-feature">{pair_label}</div>
                        <div class="sc-status" style="margin-top:4px;">{z_html}</div>
                        <div class="sc-desc" style="margin-top:6px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Sensor fusion model not available for this segment.")

        # ---- RIGHT: Yield + Recommendations ----
        with right_col:
            st.markdown('<div class="section-head">Predicted Crop Yield</div>',
                        unsafe_allow_html=True)

            pred_yield = result['predicted_yield']
            yield_min, yield_max = 2024, 5998

            fig_yield = go.Figure(go.Indicator(
                mode="number+delta+gauge",
                value=round(pred_yield, 0),
                number={'suffix': ' kg/ha',
                        'font': {'color': '#4caf72', 'size': 28, 'family': 'DM Mono'}},
                delta={'reference': 4033,
                       'increasing': {'color': '#4caf72'},
                       'decreasing': {'color': '#ef5350'},
                       'font': {'size': 14}},
                gauge={
                    'axis': {'range': [yield_min, yield_max],
                             'tickcolor': '#4a6650',
                             'tickfont': {'color': '#6b8f73', 'size': 9}},
                    'bar':  {'color': '#4caf72', 'thickness': 0.3},
                    'bgcolor': '#161d1a', 'bordercolor': '#1e3a28',
                    'steps': [
                        {'range': [yield_min, 3000], 'color': '#2d0f0f'},
                        {'range': [3000,      4500], 'color': '#0a2615'},
                        {'range': [4500, yield_max], 'color': '#0d3320'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.8, 'value': 4033
                    }
                }
            ))
            fig_yield.update_layout(
                height=240, margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e8f5e3'}
            )
            st.plotly_chart(fig_yield, use_container_width=True)

            diff     = pred_yield - 4033
            diff_txt = f"{'above' if diff >= 0 else 'below'} the regional average of 4,033 kg/ha"
            yld_col  = '#4caf72' if diff >= 0 else '#ef5350'
            yld_card = 'ok' if diff >= 0 else 'anomaly'
            st.markdown(f"""
            <div class="sensor-card {yld_card}" style="margin-top:0;">
                <div class="sc-desc">
                    Predicted yield is <strong style="color:{yld_col};">{pred_yield:.0f} kg/ha</strong>
                    — {abs(diff):.0f} kg/ha {diff_txt}.
                    {'Good growing conditions support a strong harvest.' if diff >= 0
                     else 'The detected anomalies are likely contributing to this yield reduction.'}
                </div>
            </div>""", unsafe_allow_html=True)

            # Recommendations
            st.markdown('<div class="section-head">Recommended Actions</div>',
                        unsafe_allow_html=True)

            recs = generate_recommendations(
                result['param_issues'], result['fusion_detail'],
                score, crop_sel, region_sel, pred_yield, best_thresh
            )

            badge_map = {
                'urgent':  ('rec-card urgent',  '🚨 Urgent Action'),
                'caution': ('rec-card caution', '⚡ Caution'),
                'info':    ('rec-card info',     'ℹ Info'),
            }

            for i, rec in enumerate(recs, 1):
                cls, badge = badge_map.get(rec['type'], ('rec-card info', 'ℹ Info'))
                st.markdown(f"""
                <div class="{cls}">
                    <div class="rc-num">{badge} &nbsp;·&nbsp; Action {i} of {len(recs)}</div>
                    <div class="rc-title">{rec['title']}</div>
                    <div class="rc-desc">{rec['desc']}</div>
                </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem;">
        <div style="font-size:4rem;">🌾</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem;
                    color:#6b8f73; margin-top:1rem;">
            Configure sensor values in the sidebar and click<br>
            <strong style="color:#4caf72;">Analyse Farm Conditions</strong>
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                    color:#2d4a35; margin-top:1.5rem; letter-spacing:0.1em;">
            ISOLATION FOREST · LOF · AUTOENCODER · SENSOR FUSION
        </div>
    </div>
    """, unsafe_allow_html=True)
