import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Canvas Gradebook — Refined UX", page_icon="🎓", layout="wide")

# -------------------------- Styles --------------------------
CARD_CSS = """
<style>
.kpi-card {background: var(--secondary-background-color); border-radius: 16px; padding: 16px; 
           box-shadow: 0 4px 20px rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.06);}
.kpi-title {font-size: 0.9rem; color: #93c5fd; margin-bottom: 4px;}
.kpi-value {font-size: 1.6rem; font-weight: 700; color: #e5e7eb;}
.section {margin-top: 0.75rem; margin-bottom: 0.5rem;}
.small {font-size: 0.8rem; color: #9ca3af;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {padding: 10px 16px; background: #0f172a; border-radius: 10px; 
                              border: 1px solid rgba(255,255,255,0.06);}
.stTabs [aria-selected="true"] {background: #0b2536; border-color: #0ea5e9;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# -------------------------- Helpers --------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file, dtype=str)

def detect_points_row(df):
    cand = df[df.iloc[:,0].astype(str).str.contains("Points Possible", case=False, na=False)]
    if not cand.empty:
        return cand.index[0]
    return None

def parse_numeric_series(s):
    raw = s.astype(str)
    blanks = raw.str.strip().isin(["", "nan", "NaN", "-", "—", "–"])
    excused = raw.str.strip().str.upper().isin(["EX","EXCUSED"])
    def to_num(x):
        x = str(x).strip().replace("%","")
        if x == "" or x.upper() in ["EX","EXCUSED","MI","MISSING","INC","INCOMPLETE","-","—","–"]:
            return np.nan
        try: return float(x)
        except: return np.nan
    numeric = raw.map(to_num)
    return numeric, blanks, excused

def infer_columns(df):
    cols = list(df.columns)
    meta_names = {"student","id","sis user id","sis login id","section"}
    grade_names = {
        "current score","unposted current score",
        "final score","unposted final score",
        "current grade","unposted current grade",
        "final grade","unposted final grade"
    }
    lower = {c: c.lower().strip() for c in cols}
    meta_cols = [c for c in cols if lower[c] in meta_names]
    grade_cols = [c for c in cols if lower[c] in grade_names]
    assign_cols = [c for c in cols if c not in meta_cols + grade_cols]
    return meta_cols, grade_cols, assign_cols

def compute_percent_scores(df_numeric, points_row, assign_cols):
    pct = pd.DataFrame(index=df_numeric.index, columns=assign_cols, dtype=float)
    pts = {}
    for c in assign_cols:
        try: max_pts = float(points_row[c])
        except: max_pts = np.nan
        pts[c] = max_pts
        pct[c] = (df_numeric[c] / max_pts) * 100.0 if (np.isfinite(max_pts) and max_pts > 0) else np.nan
    return pct, pd.Series(pts)

def shorten_label(s, limit=36):
    s2 = re.sub(r"\(\d+\)$", "", s).strip()
    return (s2[:limit] + "…") if len(s2) > limit else s2

def extract_category(c):
    base = re.sub(r"\(\d+\)$", "", c).strip()
    if ":" in base: return base.split(":")[0].strip()
    if "-" in base: return base.split("-")[0].strip()
    return base.split()[0] if base.split() else base

def kpi_card(title, value):
    st.markdown(f"""<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div></div>""", unsafe_allow_html=True)

def fig_layout(fig, h=420):
    fig.update_layout(height=h, margin=dict(l=30,r=10,t=40,b=35), 
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# -------------------------- Sidebar / State --------------------------
with st.sidebar:
    st.title("🎓 Gradebook")
    uploaded = st.file_uploader("Upload Canvas CSV", type=["csv"])
    st.caption("Export with **Points Possible** enabled.")

    st.markdown("**Global Controls**")
    treat_excused_as_zero = st.toggle("Excused counts as zero (include in 'including missing')", value=False)
    max_labels = st.slider("Max x-axis labels before truncation", 10, 60, 24, help="Improves readability for dense charts")
    st.divider()

# Persistent state for filters & scenarios
if "filters" not in st.session_state:
    st.session_state.filters = {"section":"All", "students":[], "categories":[]}
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}  # name -> weights dict

# -------------------------- Ingest --------------------------
if uploaded:
    raw = load_csv(uploaded)
else:
    st.info("No file uploaded — showing demo with `sample_canvas_gradebook.csv`.")
    raw = load_csv("sample_canvas_gradebook.csv")

points_idx = detect_points_row(raw)
if points_idx is None:
    points_row = pd.Series(dtype=str)
    data = raw.copy()
    has_pts = False
else:
    points_row = raw.loc[points_idx]
    data = raw.drop(index=points_idx).reset_index(drop=True)
    has_pts = True

meta_cols, grade_cols, assign_cols = infer_columns(data)

num_df = pd.DataFrame(index=data.index, columns=assign_cols, dtype=float)
is_missing = pd.DataFrame(False, index=data.index, columns=assign_cols)
is_excused = pd.DataFrame(False, index=data.index, columns=assign_cols)

for c in assign_cols:
    num, miss_blank, exc = parse_numeric_series(data[c])
    num_df[c] = num
    is_missing[c] = miss_blank | num.isna()
    is_excused[c] = exc

pct_df, points_possible = compute_percent_scores(num_df, points_row, assign_cols) if has_pts else (pd.DataFrame(), pd.Series(dtype=float))

# Canvas / computed final
final_score_col = None
for name in ["Final Score","Unposted Final Score","Current Score","Unposted Current Score"]:
    if name in data.columns:
        s, _, _ = parse_numeric_series(data[name])
        if s.notna().any():
            data[name+"_num"] = s
            final_score_col = name+"_num"
            break
if final_score_col is None and not pct_df.empty:
    data["Computed Final Score"] = pct_df.mean(axis=1)
    final_score_col = "Computed Final Score"

# Categories
categories = {c: extract_category(c) for c in assign_cols}
cat_list = sorted(set(categories.values()))

# -------------------------- Header KPIs --------------------------
st.title("Canvas Gradebook — Refined UX")
st.caption("Tabs + persistent filters • category controls • zoomable charts • scenario-based weighting")

colK1, colK2, colK3, colK4 = st.columns(4)
kpi_card("Students", data.shape[0])
kpi_card("Assignments", len(assign_cols))
if final_score_col:
    fs = data[final_score_col].astype(float)
    kpi_card("Avg Final", f"{fs.mean():.1f}%")
    kpi_card("Final < 60%", int((fs < 60).sum()))
else:
    kpi_card("Avg Final", "—")
    kpi_card("Final < 60%", "—")

st.divider()

# -------------------------- Filters Row --------------------------
fc1, fc2, fc3, fc4 = st.columns([1.2,1.2,2,1])
with fc1:
    if "Section" in data.columns:
        opts = ["All"] + sorted([x for x in data["Section"].dropna().unique().tolist() if x!=""])
        st.session_state.filters["section"] = st.selectbox("Section", opts, index=opts.index(st.session_state.filters["section"]) if st.session_state.filters["section"] in opts else 0)
with fc2:
    st.session_state.filters["categories"] = st.multiselect("Categories (filter assignments)", options=cat_list, default=st.session_state.filters["categories"])
with fc3:
    if "Student" in data.columns:
        st.session_state.filters["students"] = st.multiselect("Students (focus)", options=data["Student"].tolist(), default=st.session_state.filters["students"][:5], max_selections=8)
with fc4:
    st.markdown("<div class='section small'>Filters persist across tabs</div>", unsafe_allow_html=True)

# Apply filters
rows_mask = pd.Series(True, index=data.index)
if st.session_state.filters["section"] != "All" and "Section" in data.columns:
    rows_mask &= (data["Section"] == st.session_state.filters["section"])
data_f = data.loc[rows_mask].reset_index(drop=True)
pct_f = pct_df.loc[rows_mask] if not pct_df.empty else pct_df
missing_f = is_missing.loc[rows_mask] if not is_missing.empty else is_missing
excused_f = is_excused.loc[rows_mask] if not is_excused.empty else is_excused

assign_filtered = assign_cols[:]
if st.session_state.filters["categories"]:
    assign_filtered = [a for a in assign_cols if categories[a] in st.session_state.filters["categories"]]
if len(assign_filtered) == 0:
    assign_filtered = assign_cols[:]

# -------------------------- Tabs --------------------------
tab_overview, tab_assign, tab_students, tab_patterns, tab_weights, tab_export = st.tabs(
    ["🏠 Overview", "🧪 Assignments", "👤 Students", "🧩 Patterns", "⚖️ Weights & Scenarios", "📤 Export"]
)

# ===== Overview =====
with tab_overview:
    c1, c2 = st.columns([1.1,1])
    with c1:
        st.subheader("Final Score Distribution")
        if final_score_col:
            fig = px.histogram(data_f, x=final_score_col, nbins=12, labels={final_score_col:"Final Score (%)"})
            fig.update_traces(opacity=0.9)
            fig.update_xaxes(title="Final Score (%)")
            fig = fig_layout(fig, h=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric final score detected.")
    with c2:
        st.subheader("Grade Bands")
        grade_col = next((nm for nm in ["Final Grade","Unposted Final Grade","Current Grade","Unposted Current Grade"] if nm in data_f.columns), None)
        if grade_col:
            counts = data_f[grade_col].fillna("N/A").value_counts().reset_index()
            counts.columns = ["Grade","Count"]
            fig = px.bar(counts, x="Grade", y="Count")
            fig = fig_layout(fig, h=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No letter-grade columns found.")

    st.markdown(" ")
    if not pct_f.empty:
        st.subheader("Assignment Difficulty (Percent Averages) — Focus with category filters")
        incl = pct_f[assign_filtered].fillna(0.0).copy()
        excl = pct_f[assign_filtered].copy()
        if treat_excused_as_zero:
            pass  # excused were NaN -> now 0 after fillna
        avg_incl = incl.mean().sort_values()
        avg_excl = excl.mean().reindex(avg_incl.index)
        labels = [shorten_label(a, max_labels) for a in avg_incl.index]
        fig = go.Figure()
        fig.add_bar(x=labels, y=avg_incl.values, name="Including Missing")
        fig.add_bar(x=labels, y=avg_excl.values, name="Excluding Missing")
        fig.update_xaxes(tickangle=45)
        fig = fig_layout(fig, h=480)
        st.plotly_chart(fig, use_container_width=True)

# ===== Assignments =====
with tab_assign:
    st.subheader("Missing / Excused Heatmap")
    if not missing_f.empty:
        mat = missing_f[assign_filtered].astype(int).values.astype(float) - 0.5 * excused_f[assign_filtered].astype(int).values
        fig = px.imshow(
            mat,
            labels=dict(x="Assignments", y="Students", color="Status"),
            x=[shorten_label(a, 18) for a in assign_filtered],
            y=data_f["Student"].tolist() if "Student" in data_f.columns else None,
            aspect="auto"
        )
        fig = fig_layout(fig, h=min(800, 40 + 22*data_f.shape[0]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assignment columns detected.")

    st.subheader("Per-Assignment Summary")
    if not pct_f.empty:
        summary = pd.DataFrame({
            "Category": [categories[a] for a in assign_filtered],
            "Avg Including Missing": pct_f[assign_filtered].fillna(0.0).mean().round(1).values,
            "Avg Excluding Missing": pct_f[assign_filtered].mean().round(1).values,
            "Missing Rate": missing_f[assign_filtered].mean().round(3).values,
            "Excused Rate": excused_f[assign_filtered].mean().round(3).values,
            "Points Possible": [points_possible.get(a, np.nan) for a in assign_filtered]
        }, index=assign_filtered).sort_values("Avg Excluding Missing")
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Percent scores unavailable (no Points Possible row).")

# ===== Students =====
with tab_students:
    st.subheader("Student Trajectories (densities controlled by category filters)")
    if not pct_f.empty and "Student" in data_f.columns:
        # Choose students
        picks = st.session_state.filters["students"]
        if not picks:
            default = data_f["Student"].tolist()[:5]
            picks = st.multiselect("Pick up to 8 students", options=data_f["Student"].tolist(), default=default, max_selections=8)
            st.session_state.filters["students"] = picks
        else:
            picks = st.multiselect("Pick up to 8 students", options=data_f["Student"].tolist(), default=picks, max_selections=8)

        if picks:
            fig = go.Figure()
            for s in picks:
                row = pct_f.loc[data_f["Student"]==s, assign_filtered]
                if not row.empty:
                    fig.add_scatter(x=[shorten_label(a) for a in assign_filtered], y=row.iloc[0].values, mode="lines+markers", name=s)
            fig.update_xaxes(tickangle=45, title_text="Assignments")
            fig.update_yaxes(title_text="Score (%)", range=[0,100])
            fig = fig_layout(fig, h=460)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Student table (search/sort enabled)"):
            table = pd.concat([data_f[["Student"]] if "Student" in data_f.columns else pd.DataFrame(), pct_f.add_suffix(" [%]")], axis=1)
            st.dataframe(table, use_container_width=True)
    else:
        st.info("Need percent scores and a Student column.")

# ===== Patterns =====
with tab_patterns:
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Assignment Correlations")
        if not pct_f.empty and len(assign_filtered) >= 2:
            corr = pct_f[assign_filtered].replace(0, np.nan).corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
            fig = fig_layout(fig, h=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload a CSV with at least two assignments to compute correlations.")
    with c2:
        st.subheader("Early-Warning (First K)")
        if not pct_f.empty:
            K = st.slider("Use first K assignments", 1, len(assign_filtered), min(3, len(assign_filtered)))
            early = pct_f[assign_filtered[:K]].replace(0, np.nan).mean(axis=1)
            cutoff = np.nanpercentile(early, 20)
            flagged = early <= cutoff
            out = pd.DataFrame({
                "Student": data_f["Student"] if "Student" in data_f.columns else np.arange(len(early)),
                "Early Avg (%)": early.round(1),
                "Final (%)": data_f[final_score_col].round(1) if final_score_col else np.nan,
                "Flagged": flagged
            }).sort_values("Early Avg (%)")
            st.dataframe(out, use_container_width=True)
        else:
            st.info("Need percent scores to run early-warning.")

# ===== Weights & Scenarios =====
with tab_weights:
    st.subheader("Assignment Group Weights — What-if Analysis")
    st.caption("Set weights by category. Save scenarios and compare against baseline.")
    if len(cat_list) == 0 or pct_df.empty:
        st.info("Need percent scores and recognizable categories to run weighting.")
    else:
        cols = st.columns(min(4, len(cat_list)))
        weights = {}
        for i, cat in enumerate(cat_list):
            with cols[i % len(cols)]:
                weights[cat] = st.slider(f"{cat}", 0.0, 100.0, 100.0/len(cat_list), 1.0)
        total = sum(weights.values()) if weights else 0.0
        st.write(f"**Total = {total:.1f}%** (normalized under the hood)")

        # Compute weighted final (normalize weights to sum 1)
        if total > 0:
            wnorm = {k: v/total for k, v in weights.items()}
            # category means per student, then weighted sum
            per_cat = {}
            for cat in cat_list:
                cols_cat = [a for a in assign_cols if categories[a] == cat]
                if cols_cat:
                    per_cat[cat] = pct_f[cols_cat].replace(0, np.nan).mean(axis=1)
            # sum with weights
            wfinal = None
            for cat, series in per_cat.items():
                contrib = series * wnorm.get(cat, 0.0)
                wfinal = contrib if wfinal is None else (wfinal + contrib)
            comp = pd.DataFrame({
                "Student": data_f["Student"] if "Student" in data_f.columns else np.arange(len(pct_f)),
                "Baseline Final (%)": data_f[final_score_col].round(1) if final_score_col else pct_f.mean(axis=1).round(1),
                "What-if Final (%)": wfinal.round(1) if wfinal is not None else np.nan
            })
            st.dataframe(comp, use_container_width=True)

            # Save scenario
            scenario_name = st.text_input("Scenario name", value="My Scenario")
            if st.button("Save scenario"):
                st.session_state.scenarios[scenario_name] = weights.copy()
                st.success(f"Saved scenario '{scenario_name}'")

        # Compare scenarios
        if st.session_state.scenarios:
            st.subheader("Saved Scenarios")
            st.json(st.session_state.scenarios)

# ===== Export =====
with tab_export:
    st.subheader("Export Cleaned Percent Dataset")
    if not pct_df.empty:
        cleaned = data.copy()
        if "Student" in cleaned.columns: cleaned.set_index("Student", inplace=True)
        out = pd.concat([cleaned, pct_df.add_suffix(" [%]")], axis=1)
        st.download_button("Download cleaned CSV", data=out.to_csv().encode("utf-8"), file_name="canvas_gradebook_cleaned.csv", mime="text/csv")
    else:
        st.caption("Upload a Canvas CSV with the 'Points Possible' row to enable export.")