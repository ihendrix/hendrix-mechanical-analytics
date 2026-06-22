from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


APP_NAME = "Hendrix Mechanical Analytics"
SUBTITLE = (
    "Interactive platform for analyzing stress-strain experiments, "
    "extracting material properties, and generating research-ready outputs."
)

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")


st.markdown(
    """
    <style>
    .stApp { background:#080a0f; color:#f5f3ee; }
    .block-container { padding-top:1.8rem; padding-bottom:2rem; max-width:1420px; }
    section[data-testid="stSidebar"] { background:#11141b; border-right:1px solid rgba(255,255,255,.08); }
    section[data-testid="stSidebar"] * { color:#f5f3ee; }
    .hero-title { font-size:2.55rem; font-weight:850; letter-spacing:-.045em; margin:0; color:#fbfaf7; }
    .hero-subtitle { color:rgba(245,243,238,.70); max-width:920px; line-height:1.45; margin:.45rem 0 1.2rem; }
    .metric-card { background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.018)); border:1px solid rgba(255,255,255,.08); border-radius:20px; padding:1rem 1.1rem; min-height:104px; }
    .metric-label { font-size:.70rem; letter-spacing:.12em; text-transform:uppercase; color:rgba(245,243,238,.55); font-weight:750; margin-bottom:.8rem; }
    .metric-value { font-size:1.72rem; font-weight:850; letter-spacing:-.04em; color:#fffaf4; line-height:1; }
    .metric-sub { font-size:.80rem; color:rgba(245,243,238,.62); margin-top:.45rem; }
    .panel { background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.018)); border:1px solid rgba(255,255,255,.075); border-radius:22px; padding:1rem 1.1rem; margin:.9rem 0; }
    .section-title { font-size:1.22rem; font-weight:800; color:#fbfaf7; margin:.6rem 0 .75rem; letter-spacing:-.025em; }
    .muted { color:rgba(245,243,238,.62); font-size:.92rem; }
    .file-row { padding:.35rem .45rem; border:1px solid rgba(255,255,255,.06); border-radius:12px; margin:.25rem 0; background:rgba(255,255,255,.025); font-size:.86rem; }
    div[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.08); border-radius:16px; overflow:hidden; }
    button[kind="secondary"] { border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class TestData:
    name: str
    raw: pd.DataFrame
    clean: pd.DataFrame
    strain_col: str | None
    stress_col: str | None
    stress_unit: str
    warnings: list[str]
    status: str


def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_name(filename: str) -> str:
    name = Path(filename).stem
    name = name.replace("_corrected", "")
    name = name.replace("_", " ")
    return name.strip()


def display_name_from_index(index: int) -> str:
    return f"Specimen {index}"


def example_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    output = {}

    profiles = [
        (1.50, 1.15, "Example Control"),
        (1.55, 0.72, "Example Trial A"),
        (1.42, 1.05, "Example Trial B"),
        (1.62, 1.18, "Example Trial C"),
    ]

    for modulus, max_strain, name in profiles:
        strain = np.linspace(0, max_strain, 360)
        stress_mpa = modulus * strain + 0.05 * np.sin(strain * 9) + rng.normal(0, 0.018, len(strain))
        stress_mpa = np.maximum(stress_mpa, 0)

        drop_start = int(len(strain) * 0.94)
        stress_mpa[drop_start:] = np.linspace(
            stress_mpa[drop_start],
            stress_mpa[drop_start] * 0.25,
            len(stress_mpa[drop_start:]),
        )

        output[name] = pd.DataFrame(
            {
                "Composite strain": strain,
                "Tensile stress": stress_mpa * 1000,
            }
        )

    return output


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(data))

    sep = "\t" if suffix == ".tsv" else None
    return pd.read_csv(io.BytesIO(data), sep=sep, engine="python")


def extract_unit_row(df: pd.DataFrame):
    if df.empty:
        return df, {}

    first = df.iloc[0].astype(str).str.strip()
    numeric_ratio = pd.to_numeric(df.iloc[0], errors="coerce").notna().mean()

    has_units = first.str.contains(
        r"\(|\)|mpa|kpa|pa|mm/mm|%|n$",
        case=False,
        regex=True,
    ).mean() > 0.35

    if has_units and numeric_ratio < 0.45:
        units = {str(col): str(first.iloc[i]).strip("() ") for i, col in enumerate(df.columns)}
        return df.iloc[1:].reset_index(drop=True), units

    return df, {}


def guess_column(columns, include_terms, exclude_terms=None):
    exclude_terms = exclude_terms or []
    best = None
    best_score = -999

    for col in columns:
        low = str(col).lower()
        score = sum(term in low for term in include_terms) * 4
        score -= sum(term in low for term in exclude_terms) * 5

        if score > best_score:
            best = col
            best_score = score

    return best if best_score > 0 else None


def unit_from_column_or_row(col, units):
    text = f"{col} {units.get(col, '')}".lower()

    if "kpa" in text:
        return "kPa"
    if "mpa" in text:
        return "MPa"
    if re.search(r"\bpa\b", text):
        return "Pa"

    return "MPa"


def convert_to_mpa(series, source_unit):
    y = pd.to_numeric(series, errors="coerce")

    if source_unit == "kPa":
        return y / 1000.0
    if source_unit == "Pa":
        return y / 1_000_000.0

    return y


def smooth_series(values, method, window):
    y = values.astype(float).copy()
    window = max(3, int(window))

    if window % 2 == 0:
        window += 1

    if method == "None" or len(y) < 5:
        return y

    if method == "Moving average":
        return y.rolling(window=window, center=True, min_periods=1).mean()

    if method == "Savitzky-Golay" and savgol_filter is not None and len(y) >= window:
        return pd.Series(
            savgol_filter(y.to_numpy(), window_length=window, polyorder=2),
            index=y.index,
        )

    return y.rolling(window=window, center=True, min_periods=1).mean()


def clean_curve(df, strain_col, stress_col, units, smoothing, smooth_window, remove_outliers):
    unit = unit_from_column_or_row(stress_col, units)

    strain = pd.to_numeric(df[strain_col], errors="coerce")
    stress = convert_to_mpa(df[stress_col], unit)

    clean = (
        pd.DataFrame({"Strain": strain, "Stress_Raw_MPa": stress})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    clean = clean.sort_values("Strain").drop_duplicates("Strain").reset_index(drop=True)
    clean = clean[clean["Strain"] >= 0].copy()

    notes = []

    if clean.empty:
        return clean, unit, ["No numeric stress/strain rows detected."]

    n_base = max(5, int(len(clean) * 0.03))
    baseline = float(clean["Stress_Raw_MPa"].iloc[:n_base].median())

    clean["Stress_Corrected_MPa"] = clean["Stress_Raw_MPa"] - baseline
    notes.append(f"Baseline offset removed: {baseline:.5g} MPa")

    neg_count = int((clean["Stress_Corrected_MPa"] < 0).sum())
    clean["Stress_Corrected_MPa"] = clean["Stress_Corrected_MPa"].clip(lower=0)

    if neg_count:
        notes.append(f"Clipped {neg_count} negative stress points to zero.")

    if remove_outliers and len(clean) >= 15:
        median = clean["Stress_Corrected_MPa"].rolling(11, center=True, min_periods=1).median()
        resid = (clean["Stress_Corrected_MPa"] - median).abs()
        mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid))))
        threshold = max(0.03, 8 * mad)

        mask = resid <= threshold
        removed = int((~mask).sum())

        clean = clean[mask].reset_index(drop=True)

        if removed:
            notes.append(f"Removed {removed} spike/outlier points.")

    clean["Stress_MPa"] = smooth_series(
        clean["Stress_Corrected_MPa"],
        smoothing,
        smooth_window,
    ).clip(lower=0)

    if smoothing != "None":
        notes.append(f"Applied {smoothing.lower()} smoothing.")

    return clean, unit, notes


def detect_failure(clean):
    if clean.empty or len(clean) < 12:
        return None, "Insufficient fit region", ["Too few rows for detection."]

    stress = clean["Stress_MPa"].to_numpy()
    strain = clean["Strain"].to_numpy()

    peak_idx = int(np.nanargmax(stress))
    peak_stress = stress[peak_idx]
    peak_strain = strain[peak_idx]
    final_stress = stress[-1]
    max_strain = strain[-1]

    flags = []
    status = "Valid"
    failure_idx = peak_idx

    if peak_idx < len(stress) - 3 and peak_stress > 0:
        post = stress[peak_idx:]
        below = np.where(post <= 0.80 * peak_stress)[0]
        if len(below):
            failure_idx = peak_idx + int(below[0])

    early_end = max(8, int(len(stress) * 0.25))
    early = stress[:early_end]
    early_peak = float(np.max(early)) if len(early) else 0
    early_drops = np.diff(early)

    if early_peak > 0 and np.any(early_drops < -0.20 * early_peak):
        status = "Noisy curve"
        flags.append("Large early stress drop detected.")

    if max_strain > 0 and peak_strain < 0.35 * max_strain and peak_stress > 0:
        status = "Noisy curve" if status == "Valid" else status
        flags.append("Peak stress occurred unusually early in the strain range.")

    if peak_stress > 0 and final_stress < 0.70 * peak_stress:
        flags.append("Post-peak stress drop detected and cropped if enabled.")

    return failure_idx, status, flags


def validate_modulus(window):
    if len(window) < 5 or window["Strain"].nunique() < 2:
        return np.nan, np.nan, "Insufficient fit region"

    x = window["Strain"].to_numpy()
    y = window["Stress_MPa"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    residual_noise = np.std(y - pred) / max(np.mean(y), 1e-9)

    if slope <= 0:
        status = "Negative modulus"
    elif np.isnan(r2) or r2 < 0.75:
        status = "Low R²"
    elif residual_noise > 0.35:
        status = "Noisy curve"
    else:
        status = "Valid"

    return float(slope), float(r2), status


def calculate_metrics(clean, modulus_min, modulus_max):
    if clean.empty:
        return {
            "Peak Stress (MPa)": np.nan,
            "Strain at Peak": np.nan,
            "Young's Modulus (MPa)": np.nan,
            "Modulus R²": np.nan,
            "Modulus Fit": "Insufficient fit region",
            "Area Under Curve": np.nan,
            "Rows": 0,
        }

    peak_idx = int(clean["Stress_MPa"].idxmax())
    window = clean[(clean["Strain"] >= modulus_min) & (clean["Strain"] <= modulus_max)].copy()

    modulus, r2, fit_status = validate_modulus(window)
    auc = float(np.trapezoid(clean["Stress_MPa"], clean["Strain"])) if len(clean) >= 2 else np.nan

    return {
        "Peak Stress (MPa)": float(clean.loc[peak_idx, "Stress_MPa"]),
        "Strain at Peak": float(clean.loc[peak_idx, "Strain"]),
        "Young's Modulus (MPa)": modulus,
        "Modulus R²": r2,
        "Modulus Fit": fit_status,
        "Area Under Curve": auc,
        "Rows": int(len(clean)),
    }


def prepare_test(name, df, smoothing, smooth_window, remove_outliers, crop_failure):
    df, units = extract_unit_row(df)

    df = df.loc[:, ~pd.Series(df.columns).astype(str).str.contains("^Unnamed", regex=True).to_numpy()]
    df.columns = [str(c).strip() for c in df.columns]

    columns = list(df.columns)

    strain_col = guess_column(
        columns,
        ["composite strain", "strain", "mm/mm", "tensile strain"],
        ["stress", "load"],
    )

    stress_col = guess_column(
        columns,
        ["tensile stress", "stress", "mpa", "kpa"],
        ["strain", "load"],
    )

    numeric_cols = [
        c for c in columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 8
    ]

    if strain_col is None and numeric_cols:
        strain_col = numeric_cols[0]

    if stress_col is None and len(numeric_cols) > 1:
        stress_col = numeric_cols[1]

    warnings = []
    unit = "MPa"
    clean = pd.DataFrame()
    status = "Insufficient fit region"

    if strain_col and stress_col:
        clean, unit, warnings = clean_curve(
            df,
            strain_col,
            stress_col,
            units,
            smoothing,
            smooth_window,
            remove_outliers,
        )

        failure_idx, status, failure_notes = detect_failure(clean)
        warnings.extend(failure_notes)

        if crop_failure and failure_idx is not None and failure_idx > 5:
            clean = clean.iloc[:failure_idx + 1].copy()
            warnings.append("Curve cropped at detected failure point.")

        clean["Specimen"] = name

    else:
        warnings.append("Could not detect strain and stress columns.")

    return TestData(
        name=name,
        raw=df,
        clean=clean,
        strain_col=strain_col,
        stress_col=stress_col,
        stress_unit=unit,
        warnings=warnings,
        status=status,
    )


def make_plot(data, raw_data, show_raw):
    fig = go.Figure()

    if show_raw and raw_data is not None and not raw_data.empty:
        for name, g in raw_data.groupby("Specimen"):
            fig.add_trace(
                go.Scatter(
                    x=g["Strain"],
                    y=g["Stress_Corrected_MPa"],
                    mode="lines",
                    name=f"{name} raw",
                    opacity=0.22,
                    line=dict(width=1),
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Strain: %{x:.5f}<br>"
                        "Raw Stress: %{y:.5f} MPa"
                        "<extra></extra>"
                    ),
                )
            )

    for name, g in data.groupby("Specimen"):
        fig.add_trace(
            go.Scatter(
                x=g["Strain"],
                y=g["Stress_MPa"],
                mode="lines",
                name=name,
                line=dict(width=3),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Strain: %{x:.5f}<br>"
                    "Stress: %{y:.5f} MPa"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=610,
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        font=dict(color="#f5f3ee"),
        title="Cleaned Stress–Strain Curves",
        xaxis_title="Strain (mm/mm)",
        yaxis_title="Stress (MPa)",
        legend_title="Uploaded File",
        margin=dict(l=40, r=20, t=70, b=45),
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False, rangemode="tozero")

    return fig


with st.sidebar:
    st.header("Controls")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["csv", "xlsx", "xls", "txt", "dat", "tsv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.caption("CSV, Excel, TXT, DAT, and TSV supported.")

    st.divider()

    if uploaded_files:
        with st.expander(f"Uploaded Files ({len(uploaded_files)})"):
            for i, file in enumerate(uploaded_files, start=1):
                st.write(f"✓ {display_name_from_index(i)}")
    else:
        st.caption("No files uploaded. Demo data is shown.")

    st.divider()

    smoothing = st.selectbox(
        "Smoothing",
        ["Savitzky-Golay", "Moving average", "None"],
        index=0,
    )

    smooth_window = st.slider(
        "Smoothing window",
        min_value=5,
        max_value=51,
        value=17,
        step=2,
    )

    remove_outliers = st.checkbox("Remove spike outliers", value=True)
    crop_failure = st.checkbox("Crop after peak/failure", value=True)
    show_raw = st.checkbox("Show raw overlay", value=False)

    st.divider()
    st.caption("Modulus fit region")

    modulus_min = st.number_input(
        "Start strain",
        value=0.005,
        min_value=0.0,
        step=0.005,
        format="%.4f",
    )

    modulus_max = st.number_input(
        "End strain",
        value=0.080,
        min_value=0.0,
        step=0.005,
        format="%.4f",
    )


st.markdown(
    f'<div class="hero-title">{APP_NAME}</div>'
    f'<div class="hero-subtitle">{SUBTITLE}</div>',
    unsafe_allow_html=True,
)


raw_frames = {}

if uploaded_files:
    used_names = {}

    for uploaded in uploaded_files:
        try:
            base = safe_name(uploaded.name)

            if base in used_names:
                used_names[base] += 1
                display_name = f"{base} ({used_names[base]})"
            else:
                used_names[base] = 1
                display_name = base

            raw_frames[display_name] = read_uploaded_file(uploaded)

        except Exception as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")

else:
    st.info("Upload files to analyze your own data. Showing dummy example data for layout preview.")
    raw_frames = example_data()


tests = [
    prepare_test(
        name,
        df,
        smoothing,
        smooth_window,
        remove_outliers,
        crop_failure,
    )
    for name, df in raw_frames.items()
]

valid_tests = [t for t in tests if not t.clean.empty]

if not valid_tests:
    st.error("No usable stress-strain data detected. Check for numeric strain and stress columns.")
    st.stop()


with st.sidebar:
    st.divider()

    names = [t.name for t in valid_tests]

    if "selected_specimens" not in st.session_state or set(st.session_state.selected_specimens) - set(names):
        st.session_state.selected_specimens = names

    c1, c2 = st.columns(2)

    if c1.button("Select All", use_container_width=True):
        st.session_state.selected_specimens = names

    if c2.button("Clear All", use_container_width=True):
        st.session_state.selected_specimens = []

    selected_names = st.multiselect(
        "Files to plot",
        options=names,
        key="selected_specimens",
        label_visibility="collapsed",
    )


selected_tests = [t for t in valid_tests if t.name in selected_names]

all_clean = (
    pd.concat([t.clean for t in selected_tests], ignore_index=True)
    if selected_tests
    else pd.DataFrame()
)

raw_overlay = all_clean.copy()

metrics = []

for t in selected_tests:
    m = calculate_metrics(t.clean, modulus_min, modulus_max)
    m.update(
        {
            "File": t.name,
            "Detected Strain Column": t.strain_col,
            "Detected Stress Column": t.stress_col,
        }
    )
    metrics.append(m)

metrics_df = pd.DataFrame(metrics)

mean_modulus = metrics_df["Young's Modulus (MPa)"].mean() if not metrics_df.empty else np.nan
max_stress = metrics_df["Peak Stress (MPa)"].max() if not metrics_df.empty else np.nan
total_rows = int(all_clean.shape[0]) if not all_clean.empty else 0


m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card("Files Plotted", len(selected_tests), "Selected for analysis")

with m2:
    metric_card("Clean Rows", total_rows, "After parsing and cleaning")

with m3:
    metric_card("Max Stress", "—" if pd.isna(max_stress) else f"{max_stress:.3f}", "MPa")

with m4:
    metric_card("Mean Modulus", "—" if pd.isna(mean_modulus) else f"{mean_modulus:.3f}", "MPa")

st.caption(
    "Displayed values depend on uploaded file units and selected fit region. "
    "Validate units before interpreting material properties."
)


st.markdown('<div class="section-title">Stress–Strain Analysis</div>', unsafe_allow_html=True)

if all_clean.empty:
    st.warning("Select at least one file.")
    fig = go.Figure()
else:
    fig = make_plot(all_clean, raw_overlay, show_raw)
    st.plotly_chart(fig, use_container_width=True)


st.markdown('<div class="section-title">Material Property Summary</div>', unsafe_allow_html=True)

st.caption(
    "Needs Review indicates low modulus-fit confidence, negative or near-zero modulus, "
    "excessive noise, or insufficient linear-region quality."
)

if not metrics_df.empty:
    display = metrics_df[
        [
            "File",
            "Peak Stress (MPa)",
            "Strain at Peak",
            "Young's Modulus (MPa)",
            "Modulus R²",
            "Modulus Fit",
            "Area Under Curve",
            "Rows",
        ]
    ].copy()

    for col in [
        "Peak Stress (MPa)",
        "Strain at Peak",
        "Young's Modulus (MPa)",
        "Modulus R²",
        "Area Under Curve",
    ]:
        display[col] = display[col].round(5)

    st.dataframe(display, use_container_width=True, hide_index=True)

    bar = px.bar(
        display,
        x="File",
        y="Young's Modulus (MPa)",
        title="Young's Modulus by File",
        hover_data=["Modulus Fit", "Modulus R²", "Peak Stress (MPa)"],
    )

    bar.update_layout(
        template="plotly_dark",
        height=360,
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        font=dict(color="#f5f3ee"),
        margin=dict(l=40, r=20, t=60, b=90),
    )

    bar.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    bar.update_yaxes(gridcolor="rgba(255,255,255,.07)")

    st.plotly_chart(bar, use_container_width=True)


with st.expander("Cleaning notes", expanded=False):
    for t in selected_tests:
        st.write(f"**{t.name}**")

        for note in t.warnings[:8]:
            st.write(f"- {note}")

        st.write(f"Detected columns: `{t.strain_col}` and `{t.stress_col}`")


with st.expander("Cleaned data preview + downloads", expanded=False):
    st.dataframe(all_clean, use_container_width=True, hide_index=True)

    cleaned_csv = all_clean.to_csv(index=False).encode("utf-8") if not all_clean.empty else b""
    summary_csv = metrics_df.to_csv(index=False).encode("utf-8") if not metrics_df.empty else b""
    chart_html = fig.to_html(include_plotlyjs="cdn") if not all_clean.empty else ""

    a, b, c = st.columns(3)

    with a:
        st.download_button(
            "Download cleaned CSV",
            cleaned_csv,
            "cleaned_stress_strain_data.csv",
            "text/csv",
            use_container_width=True,
        )

    with b:
        st.download_button(
            "Download summary CSV",
            summary_csv,
            "mechanical_summary.csv",
            "text/csv",
            use_container_width=True,
        )

    with c:
        st.download_button(
            "Download chart HTML",
            chart_html,
            "stress_strain_analysis.html",
            "text/html",
            use_container_width=True,
        )


st.markdown(
    '<div class="panel"><b>Notes</b><br>'
    '<span class="muted">This dashboard converts uploaded mechanical testing files into cleaned '
    'stress-strain curves, extracted material properties, quality flags, and downloadable outputs. '
    'It is designed as a public-safe demonstration of an automated research workflow, not as a '
    'substitute for final materials validation.</span></div>',
    unsafe_allow_html=True,
)
