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
import plotly.express as px
from plotly.subplots import make_subplots

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

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main { background-color: #0d1117; }

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1e12 50%, #0d1117 100%);
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #e8f5e3;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #4caf72;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: #161d1a;
    border: 1px solid #1e3a28;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #4caf72;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #6b8f73;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* Result box */
.result-anomaly {
    background: linear-gradient(135deg, #2d0f0f, #1a0808);
    border: 1.5px solid #e53935;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-normal {
    background: linear-gradient(135deg, #0a2615, #061a0e);
    border: 1.5px solid #4caf72;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-borderline {
    background: linear-gradient(135deg, #2d2200, #1a1500);
    border: 1.5px solid #f5a623;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.result-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #9ab5a0;
    margin-top: 0.3rem;
}

/* Section header */
.section-head {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4caf72;
    border-bottom: 1px solid #1e3a28;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Feature issue tags */
.issue-tag {
    display: inline-block;
    background: #2d1a00;
    border: 1px solid #f5a623;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #f5c842;
    margin: 3px 3px 3px 0;
    font-family: 'DM Mono', monospace;
}
.ok-tag {
    display: inline-block;
    background: #0a2615;
    border: 1px solid #4caf72;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #6fcf97;
    margin: 3px 3px 3px 0;
    font-family: 'DM Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1a12;
    border-right: 1px solid #1e3a28;
}

/* Inputs */
.stSlider > div > div > div > div {
    background: #4caf72 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_bundle():
    with open("models/model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    # Load Keras autoencoders
    autoencoders = {}
    for fname in os.listdir("models"):
        if fname.startswith("ae_") and fname.endswith(".keras"):
            parts   = fname.replace(".keras", "").split("_")
            crop    = int(parts[1])
            region  = int(parts[2])
            autoencoders[(crop, region)] = load_model(f"models/{fname}", compile=False)

    bundle['autoencoders'] = autoencoders
    return bundle

bundle = load_bundle()

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
# SCORE HELPERS
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

def predict(user_data):
    crop_enc   = encode_val(user_data['crop_type'],   encoders['crop_type'])
    region_enc = encode_val(user_data['region'],       encoders['region'])
    if crop_enc is None or region_enc is None:
        return None
    key = (crop_enc, region_enc)
    if key not in if_models:
        return None

    X = np.array([[user_data[f] for f in FEATURES]])

    if_raw   = get_if_score(if_models[key], X)
    lof_raw  = get_lof_score(lof_models[key], X)
    ae_raw   = get_ae_score(autoencoders[key], scalers[key], X)
    fs_raw, fusion_detail = get_fusion_score(user_data, key)

    if_s   = float(norm_if.transform([[if_raw]])[0][0])
    lof_s  = float(norm_lof.transform([[lof_raw]])[0][0])
    ae_s   = float(norm_ae.transform([[ae_raw]])[0][0])
    fus_s  = float(norm_fus.transform([[fs_raw]])[0][0])

    final_score = (weights['if']     * if_s  +
                   weights['lof']    * lof_s +
                   weights['ae']     * ae_s  +
                   weights['fusion'] * fus_s)

    ae_feat = get_ae_feature_scores(autoencoders[key], scalers[key], X)

    # Parameter range check
    seg_s = segment_stats.get(key, {})
    param_issues = []
    for f in FEATURES:
        if f not in seg_s:
            continue
        val = user_data[f]
        low, high = seg_s[f]['low'], seg_s[f]['high']
        if val < low:
            param_issues.append({'feature': f, 'value': val,
                                  'status': 'LOW', 'expected': f"> {low:.2f}"})
        elif val > high:
            param_issues.append({'feature': f, 'value': val,
                                  'status': 'HIGH', 'expected': f"< {high:.2f}"})

    # Yield prediction
    extra = {
        'sunlight_hours':     user_data.get('sunlight_hours', 7.0),
        'pesticide_usage_ml': user_data.get('pesticide_usage_ml', 25.0),
        'total_days':         user_data.get('total_days', 120),
        'season_sin':         user_data.get('season_sin', 0.0),
        'season_cos':         user_data.get('season_cos', 1.0),
        'crop_type':          crop_enc,
        'region':             region_enc,
    }
    yield_input = np.array([[user_data[f] for f in FEATURES] +
                             [extra[k] for k in ['sunlight_hours','pesticide_usage_ml',
                                                  'total_days','season_sin','season_cos',
                                                  'crop_type','region']]])
    predicted_yield = float(yield_model.predict(yield_input)[0])

    return {
        'final_score':    final_score,
        'if_s': if_s, 'lof_s': lof_s, 'ae_s': ae_s, 'fus_s': fus_s,
        'fusion_detail':  fusion_detail,
        'ae_feat':        ae_feat,
        'param_issues':   param_issues,
        'predicted_yield':predicted_yield,
        'seg_stats':      seg_s,
        'key':            key,
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem;">🌾 AgroSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Farm Context")
    crop_sel   = st.selectbox("Crop Type", CROP_LIST)
    region_sel = st.selectbox("Region",    REGION_LIST)

    st.markdown("### Sensor Readings")
    soil_moist = st.slider("Soil Moisture (%)",  0.0,  60.0, 25.0, 0.1)
    soil_ph    = st.slider("Soil pH",            4.0,   9.0,  6.5, 0.01)
    temperature= st.slider("Temperature (°C)",  10.0,  55.0, 24.0, 0.1)
    rainfall   = st.slider("Rainfall (mm)",      0.0, 400.0,180.0, 1.0)
    humidity   = st.slider("Humidity (%)",       0.0, 100.0, 65.0, 0.1)
    ndvi       = st.slider("NDVI Index",         0.0,   1.0,  0.6, 0.01)

    st.markdown("### Farm Details")
    sunlight   = st.slider("Sunlight Hours/day", 2.0,  14.0,  7.0, 0.1)
    pesticide  = st.slider("Pesticide (ml)",     0.0,  60.0, 25.0, 0.5)
    total_days = st.slider("Growing Days",       60,   180,  120,  1)
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
st.markdown('<div class="hero-sub">Hybrid Contextual Anomaly Detection · Precision Agriculture</div>', unsafe_allow_html=True)

# Model performance strip
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
            status_html = f"""
            <div class="result-anomaly">
                <div class="result-title" style="color:#ef5350;">⚠ Anomaly Detected</div>
                <div class="result-score">Ensemble Score: {score:.3f}  |  Threshold: {best_thresh:.3f}</div>
                <div style="font-size:0.85rem; color:#ef9a9a; margin-top:0.5rem;">
                Abnormal farming conditions identified. Review parameter and sensor analysis below.
                </div>
            </div>"""
        elif score >= 0.5:
            status_html = f"""
            <div class="result-borderline">
                <div class="result-title" style="color:#f5a623;">⚡ Borderline Conditions</div>
                <div class="result-score">Ensemble Score: {score:.3f}  |  Threshold: {best_thresh:.3f}</div>
                <div style="font-size:0.85rem; color:#ffe082; margin-top:0.5rem;">
                Readings approaching anomalous range. Monitor closely.
                </div>
            </div>"""
        else:
            status_html = f"""
            <div class="result-normal">
                <div class="result-title" style="color:#66bb6a;">✓ Conditions Normal</div>
                <div class="result-score">Ensemble Score: {score:.3f}  |  Threshold: {best_thresh:.3f}</div>
                <div style="font-size:0.85rem; color:#a5d6a7; margin-top:0.5rem;">
                All sensor readings within expected range for {crop_sel} in {region_sel}.
                </div>
            </div>"""

        st.markdown(status_html, unsafe_allow_html=True)

        # ============================================================
        # TWO COLUMN LAYOUT: Charts left, Analysis right
        # ============================================================
        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            # ---- Gauge Chart ----
            st.markdown('<div class="section-head">Ensemble Anomaly Score</div>', unsafe_allow_html=True)
            gauge_color = "#ef5350" if score > best_thresh else ("#f5a623" if score >= 0.5 else "#4caf72")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score, 3),
                number={'font': {'color': gauge_color, 'size': 36, 'family': 'DM Mono'}},
                gauge={
                    'axis': {'range': [0, 1], 'tickcolor': '#4a6650',
                             'tickfont': {'color': '#6b8f73', 'size': 11}},
                    'bar':  {'color': gauge_color, 'thickness': 0.25},
                    'bgcolor': '#161d1a',
                    'bordercolor': '#1e3a28',
                    'steps': [
                        {'range': [0, best_thresh], 'color': '#0a2615'},
                        {'range': [best_thresh, 1], 'color': '#2d0f0f'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.8,
                        'value': best_thresh
                    }
                }
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e8f5e3'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ---- Model Score Breakdown ----
            st.markdown('<div class="section-head">Model Score Breakdown</div>', unsafe_allow_html=True)
            model_names  = ['IF', 'LOF', 'AE', 'Fusion']
            model_scores = [result['if_s'], result['lof_s'],
                            result['ae_s'], result['fus_s']]
            model_weights= [weights['if'], weights['lof'],
                            weights['ae'], weights['fusion']]
            bar_cols_models = ['#5DCAA5', '#1D9E75', '#0F6E56', '#378ADD']

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=model_names,
                y=model_scores,
                marker_color=bar_cols_models,
                text=[f"{s:.3f}" for s in model_scores],
                textposition='outside',
                textfont={'color': '#e8f5e3', 'size': 11},
                name='Score',
                customdata=[[f"weight={w:.3f}"] for w in model_weights],
                hovertemplate='%{x}<br>Score: %{y:.3f}<br>%{customdata[0]}<extra></extra>'
            ))
            fig_bar.add_hline(y=0.5, line_dash="dot",
                              line_color="#f5a623", line_width=1,
                              annotation_text="0.5", annotation_font_color="#f5a623")
            fig_bar.update_layout(
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis={'tickfont': {'color': '#9ab5a0'}, 'gridcolor': '#1e3a28'},
                yaxis={'range': [0, 1.2], 'tickfont': {'color': '#9ab5a0'},
                       'gridcolor': '#1e3a28'},
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ---- Sensor Readings vs Normal Range ----
            st.markdown('<div class="section-head">Sensor Readings vs Normal Range</div>', unsafe_allow_html=True)
            seg_s   = result['seg_stats']
            f_names = [f.replace('_', ' ').replace('%', '').strip() for f in FEATURES]
            f_vals  = [user_data[f] for f in FEATURES]
            f_low   = [seg_s[f]['low']  if f in seg_s else 0 for f in FEATURES]
            f_high  = [seg_s[f]['high'] if f in seg_s else 1 for f in FEATURES]
            f_mean  = [seg_s[f]['mean'] if f in seg_s else 0.5 for f in FEATURES]

            # Normalize each feature 0-1 for radar
            f_norm_val  = []
            f_norm_low  = []
            f_norm_high = []
            f_norm_mean = []
            for i, f in enumerate(FEATURES):
                rng = f_high[i] - f_low[i] + 1e-6
                f_norm_val.append((f_vals[i] - f_low[i]) / rng)
                f_norm_low.append(0.0)
                f_norm_high.append(1.0)
                f_norm_mean.append(0.5)

            dot_colors = []
            for i, f in enumerate(FEATURES):
                if f_vals[i] < f_low[i] or f_vals[i] > f_high[i]:
                    dot_colors.append('#ef5350')
                else:
                    dot_colors.append('#4caf72')

            fig_range = go.Figure()
            # Normal band
            fig_range.add_trace(go.Bar(
                name='Normal Range',
                x=f_names,
                y=[1.0]*len(FEATURES),
                marker_color='#1e3a28',
                hoverinfo='skip'
            ))
            # Actual value dots
            fig_range.add_trace(go.Scatter(
                name='Your Reading',
                x=f_names,
                y=f_norm_val,
                mode='markers',
                marker=dict(color=dot_colors, size=12, symbol='diamond'),
                hovertemplate='%{x}<br>Value: %{customdata:.2f}<extra></extra>',
                customdata=f_vals
            ))
            fig_range.update_layout(
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=30),
                xaxis={'tickfont': {'color': '#9ab5a0', 'size': 9},
                       'gridcolor': '#1e3a28'},
                yaxis={'visible': False},
                legend={'font': {'color': '#9ab5a0'}, 'bgcolor': 'rgba(0,0,0,0)'},
                barmode='overlay'
            )
            st.plotly_chart(fig_range, use_container_width=True)

        with right_col:

            # ---- Parameter Analysis ----
            st.markdown('<div class="section-head">Parameter Analysis</div>', unsafe_allow_html=True)
            issues = result['param_issues']
            if issues:
                for iss in issues:
                    st.markdown(f"""
                    <span class="issue-tag">
                        {iss['feature'].replace('_',' ')} = {iss['value']:.2f}
                        &nbsp;{iss['status']}&nbsp;
                        (expected {iss['expected']})
                    </span>""", unsafe_allow_html=True)
            else:
                st.markdown('<span class="ok-tag">✓ All parameters within normal range</span>',
                            unsafe_allow_html=True)

            # ---- AE Feature Reconstruction Error ----
            st.markdown('<div class="section-head">Autoencoder Feature Anomaly Scores</div>', unsafe_allow_html=True)
            ae_feat = result['ae_feat']
            ae_sorted = sorted(ae_feat.items(), key=lambda x: x[1], reverse=True)
            ae_names  = [f[0].replace('_', ' ') for f, _ in ae_sorted]
            ae_vals   = [v for _, v in ae_sorted]
            max_ae    = max(ae_vals) + 1e-9
            ae_colors = ['#ef5350' if v/max_ae > 0.6 else
                         '#f5a623' if v/max_ae > 0.3 else '#4caf72' for v in ae_vals]

            fig_ae = go.Figure(go.Bar(
                x=ae_vals, y=ae_names, orientation='h',
                marker_color=ae_colors,
                text=[f"{v:.5f}" for v in ae_vals],
                textposition='outside',
                textfont={'color': '#9ab5a0', 'size': 9},
                hovertemplate='%{y}<br>Error: %{x:.5f}<extra></extra>'
            ))
            fig_ae.update_layout(
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis={'tickfont': {'color': '#9ab5a0', 'size': 9},
                       'gridcolor': '#1e3a28', 'title': 'Reconstruction Error'},
                yaxis={'tickfont': {'color': '#9ab5a0', 'size': 9}},
                showlegend=False,
            )
            st.plotly_chart(fig_ae, use_container_width=True)

            # ---- Sensor Fusion ----
            st.markdown('<div class="section-head">Learned Sensor Fusion (z-scores)</div>', unsafe_allow_html=True)
            fd = result['fusion_detail']
            if fd:
                fus_pairs  = list(fd.keys())
                fus_zscores= list(fd.values())
                fus_colors = ['#ef5350' if z > 2.0 else
                              '#f5a623' if z > 1.0 else '#4caf72' for z in fus_zscores]
                fig_fus = go.Figure(go.Bar(
                    x=fus_zscores, y=fus_pairs, orientation='h',
                    marker_color=fus_colors,
                    text=[f"{z:.2f}σ" for z in fus_zscores],
                    textposition='outside',
                    textfont={'color': '#9ab5a0', 'size': 10},
                    hovertemplate='%{y}<br>z-score: %{x:.2f}<extra></extra>'
                ))
                fig_fus.add_vline(x=2.0, line_dash="dot",
                                  line_color="#ef5350", line_width=1.5,
                                  annotation_text="2σ threshold",
                                  annotation_font_color="#ef5350")
                fig_fus.update_layout(
                    height=200,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=60, t=10, b=10),
                    xaxis={'tickfont': {'color': '#9ab5a0'},
                           'gridcolor': '#1e3a28', 'title': 'z-score'},
                    yaxis={'tickfont': {'color': '#9ab5a0', 'size': 9}},
                    showlegend=False,
                )
                st.plotly_chart(fig_fus, use_container_width=True)
            else:
                st.info("No fusion model available for this segment.")

            # ---- Yield Prediction ----
            st.markdown('<div class="section-head">Predicted Crop Yield</div>', unsafe_allow_html=True)
            pred_yield = result['predicted_yield']
            # Typical range from training data
            yield_min, yield_max = 2024, 5998
            yield_pct = int(100 * (pred_yield - yield_min) / (yield_max - yield_min))

            fig_yield = go.Figure(go.Indicator(
                mode="number+delta+gauge",
                value=round(pred_yield, 0),
                number={'suffix': ' kg/ha',
                        'font': {'color': '#4caf72', 'size': 28, 'family': 'DM Mono'}},
                delta={'reference': 4033, 'increasing': {'color': '#4caf72'},
                       'decreasing': {'color': '#ef5350'},
                       'font': {'size': 14}},
                gauge={
                    'axis': {'range': [yield_min, yield_max],
                             'tickcolor': '#4a6650',
                             'tickfont': {'color': '#6b8f73', 'size': 9}},
                    'bar': {'color': '#4caf72', 'thickness': 0.3},
                    'bgcolor': '#161d1a',
                    'bordercolor': '#1e3a28',
                    'steps': [
                        {'range': [yield_min, 3000], 'color': '#2d0f0f'},
                        {'range': [3000, 4500],      'color': '#0a2615'},
                        {'range': [4500, yield_max], 'color': '#0d3320'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.8, 'value': 4033
                    }
                }
            ))
            fig_yield.add_annotation(
                text="↑ above avg" if pred_yield > 4033 else "↓ below avg",
                x=0.5, y=0.15, showarrow=False,
                font={'color': '#4caf72' if pred_yield > 4033 else '#ef5350',
                      'size': 11, 'family': 'DM Mono'},
                xref='paper', yref='paper'
            )
            fig_yield.update_layout(
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=10),
                font={'color': '#e8f5e3'}
            )
            st.plotly_chart(fig_yield, use_container_width=True)

        # ---- Model Weight Table ----
        st.markdown('<div class="section-head">Ensemble Model Weights (F1-based)</div>', unsafe_allow_html=True)
        w_df = pd.DataFrame([
            {'Model': 'Isolation Forest', 'F1-based Weight': round(weights['if'], 3),
             'Score (this input)': round(result['if_s'], 3)},
            {'Model': 'Local Outlier Factor', 'F1-based Weight': round(weights['lof'], 3),
             'Score (this input)': round(result['lof_s'], 3)},
            {'Model': 'Autoencoder', 'F1-based Weight': round(weights['ae'], 3),
             'Score (this input)': round(result['ae_s'], 3)},
            {'Model': 'Sensor Fusion', 'F1-based Weight': round(weights['fusion'], 3),
             'Score (this input)': round(result['fus_s'], 3)},
        ])
        st.dataframe(w_df, use_container_width=True, hide_index=True)

else:
    # ---- Landing state ----
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #4a6650;">
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
