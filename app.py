import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score
)

st.set_page_config(page_title="Flood Analytics Dashboard", page_icon="🌧️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background-color: #f1f5f9; }
.dashboard-title { text-align:center; color:#0f172a; font-size:2.25rem; font-weight:700; margin-top:1rem; }
.dashboard-subtitle { text-align:center; color:#0d9488; font-size:1rem; font-weight:600; margin-bottom:1.5rem; letter-spacing:1px; }
.metric-box { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; border-top:4px solid #0d9488; }
.metric-box h3 { color:#64748b; font-size:.8rem; margin:0 0 8px; text-transform:uppercase; }
.metric-box h2 { color:#0f172a; font-size:1.7rem; margin:0; }
.insight-text { background:#fff; border-left:4px solid #0d9488; padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:12px; }
[data-testid="stSidebar"] { background:#fff; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="dashboard-title">🌧️ Flood Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">HISTORICAL FLOOD ANALYSIS & DATA-DRIVEN PREPAREDNESS</div>', unsafe_allow_html=True)
st.caption("Study area: Bunawan, Agusan del Sur | Research Objectives: (1) describe flood frequency, severity and seasonality; (2) identify patterns using K-Means, Decision Tree and Apriori; (3) derive evidence-based preparedness recommendations.")

MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

def clean_text(series):
    return series.astype("string").str.replace("\u00a0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

def number_from(series):
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
        errors="coerce"
    )

def preprocess(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Remove fully empty columns (the supplied workbook contains an unused final column).
    df = df.dropna(axis=1, how="all")
    for c in ["Month", "Municipality", "Barangay", "Flood Cause", "Weather", "Temperature(°C)", "Wind", "Pressure"]:
        if c in df.columns:
            df[c] = clean_text(df[c]).replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
    # Date fields are sparsely populated in the supplied workbook; forward-fill year/month only.
    for c in ["Year", "Month", "Municipality", "Flood Cause"]:
        if c in df.columns:
            df[c] = df[c].ffill()
    out = pd.DataFrame(index=df.index)
    out["Year"] = pd.to_numeric(df.get("Year", pd.Series(index=df.index, dtype=float)), errors="coerce")
    out["Month"] = df.get("Month", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str).str.title()
    out["Day"] = pd.to_numeric(df.get("Day", pd.Series(index=df.index, dtype=float)), errors="coerce")
    out["Municipality"] = df.get("Municipality", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    out["Barangay"] = df.get("Barangay", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str).str.title()
    out["Flood_Occurrence"] = pd.to_numeric(df.get("Flood Occurrence", pd.Series(index=df.index, dtype=float)), errors="coerce")
    out["Flood_Cause"] = df.get("Flood Cause", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str).str.strip()
    out["Water_Level_Ft"] = number_from(df.get("Water Level", pd.Series(index=df.index, dtype=object)))
    out["Families_Affected"] = number_from(df.get("No. of Families affected", pd.Series(index=df.index, dtype=object)))
    out["Damage_Infrastructure"] = number_from(df.get("Damage Infrastructure", pd.Series(index=df.index, dtype=object)))
    out["Damage_Agriculture"] = number_from(df.get("Damage Agriculture", pd.Series(index=df.index, dtype=object)))
    out["Total_Damage"] = out["Damage_Infrastructure"].fillna(0) + out["Damage_Agriculture"].fillna(0)
    out["Weather"] = df.get("Weather", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    out["Temperature"] = df.get("Temperature(°C)", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    out["Rainfall_mm"] = pd.to_numeric(df.get("Rainfall (mm)", pd.Series(index=df.index, dtype=float)), errors="coerce")
    out["Wind"] = df.get("Wind", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    out["Pressure"] = df.get("Pressure", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    out = out[out["Barangay"].ne("Unknown") & out["Barangay"].ne("")].copy()
    out["Month_Num"] = out["Month"].map({m:i+1 for i,m in enumerate(MONTHS)})
    out["Season"] = out["Month_Num"].apply(lambda m: "Dry Season" if m in [11,12,1,2,3,4] else ("Wet Season" if m in [5,6,7,8,9,10] else "Unknown"))

    def severity(r):
        high = 0
        low = 0
        wl, fam, dmg = r["Water_Level_Ft"], r["Families_Affected"], r["Total_Damage"]
        if pd.notna(wl):
            high += wl > 4
            low += wl <= 2
        if pd.notna(fam):
            high += fam > 200
            low += fam < 50
        if pd.notna(dmg):
            high += dmg > 500000
            low += dmg < 100000
        if high >= 2: return "High"
        if low == 3: return "Low"
        return "Moderate"
    out["Severity"] = out.apply(severity, axis=1)
    return out

def money(v):
    return f"₱{v:,.0f}"

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload flood dataset (Excel/CSV)", type=["xlsx","xls","csv"])
    st.caption("The included Excel dataset is loaded automatically when available.")
    default_path = "FloodData 3 new.xlsx"

raw = None
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    try:
        raw = pd.read_excel(default_path)
    except Exception:
        st.warning("Upload your Excel or CSV dataset in the sidebar to start.")
        st.stop()

df = preprocess(raw)
if df.empty:
    st.error("No usable Barangay records were found. Check the dataset column names.")
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Filters")
    brgys = sorted(df["Barangay"].dropna().unique().tolist())
    selected_brgys = st.multiselect("Barangay", brgys, default=brgys)
    valid_years = df["Year"].dropna()
    if not valid_years.empty:
        y_min, y_max = int(valid_years.min()), int(valid_years.max())
        if y_min < y_max:
            yrange = st.slider("Year range", y_min, y_max, (y_min, y_max))
        else:
            yrange = (y_min, y_max)
    else:
        yrange = None

filtered = df[df["Barangay"].isin(selected_brgys)].copy()
if yrange:
    filtered = filtered[filtered["Year"].between(yrange[0], yrange[1])]
if filtered.empty:
    st.warning("No records match the selected filters. Please adjust your filters.")
    st.stop()

tabs = st.tabs(["Overview", "Flood Frequency & Seasonality", "Weather & Damage", "K-Means Risk Clustering", "Decision Tree Severity", "Apriori Association Rules", "Preparedness Recommendations"])

with tabs[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><h3>Total Records</h3><h2>{len(filtered):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><h3>Barangays</h3><h2>{filtered["Barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><h3>Families Affected</h3><h2>{filtered["Families_Affected"].sum():,.0f}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><h3>Total Recorded Damage</h3><h2>{money(filtered["Total_Damage"].sum())}</h2></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="insight-text">The filtered dataset contains <b>{len(filtered):,}</b> records across <b>{filtered["Barangay"].nunique()}</b> barangays. Recorded affected families total <b>{filtered["Families_Affected"].sum():,.0f}</b>, and combined recorded infrastructure and agricultural damage totals <b>{money(filtered["Total_Damage"].sum())}</b>.</div>', unsafe_allow_html=True)
    left,right = st.columns(2)
    with left:
        counts = filtered["Barangay"].value_counts()
        fig,ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#0d9488")
        ax.set_title("Records by Barangay"); ax.set_xlabel(""); ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=45); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        causes = filtered["Flood_Cause"].replace("Unknown", np.nan).dropna().value_counts().head(8)
        if len(causes):
            fig,ax = plt.subplots(figsize=(8,4))
            causes.plot(kind="pie", autopct="%1.1f%%", ax=ax, ylabel="")
            ax.set_title("Flood Cause Distribution"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("No flood-cause information is available for the selected records.")
    st.subheader("Preview of cleaned data")
    st.dataframe(filtered[["Year","Month","Municipality","Barangay","Flood_Cause","Water_Level_Ft","Families_Affected","Total_Damage","Rainfall_mm","Severity"]], use_container_width=True)

with tabs[1]:
    left,right = st.columns(2)
    with left:
        causes = filtered["Flood_Cause"].replace("Unknown", np.nan).dropna().value_counts().head(12)
        if len(causes):
            fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(y=causes.index,x=causes.values,ax=ax,color="#0d9488")
            ax.set_title("Most Common Flood Causes"); ax.set_xlabel("Records"); ax.set_ylabel("")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("No flood-cause data is available.")
    with right:
        bcounts=filtered["Barangay"].value_counts().head(15)
        fig,ax=plt.subplots(figsize=(8,5)); sns.barplot(y=bcounts.index,x=bcounts.values,ax=ax,color="#64748b")
        ax.set_title("Barangays by Record Count"); ax.set_xlabel("Records"); ax.set_ylabel("")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if filtered["Flood_Cause"].replace("Unknown",np.nan).notna().any():
        cause=filtered["Flood_Cause"].replace("Unknown",np.nan).mode().iloc[0]
        st.markdown(f'<div class="insight-text">The most frequently recorded flood cause in the current selection is <b>{cause}</b>.</div>',unsafe_allow_html=True)

with tabs[2]:
    a,b=st.columns(2)
    with a:
        month_counts=filtered["Month"].value_counts().reindex(MONTHS).fillna(0)
        fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(x=month_counts.index,y=month_counts.values,ax=ax,color="#3b82f6")
        ax.set_title("Records by Month"); ax.set_ylabel("Records"); ax.tick_params(axis="x",rotation=45)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        season_counts=filtered["Season"].value_counts()
        fig,ax=plt.subplots(figsize=(7,4)); season_counts.plot(kind="bar",ax=ax,color=["#0d9488","#f59e0b","#94a3b8"])
        ax.set_title("Dry vs Wet Season"); ax.set_ylabel("Records"); ax.tick_params(axis="x",rotation=0)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    weather=filtered.dropna(subset=["Rainfall_mm"])
    if not weather.empty:
        fig,ax=plt.subplots(figsize=(9,4))
        sns.scatterplot(data=weather,x="Rainfall_mm",y="Water_Level_Ft",hue="Season",ax=ax)
        ax.set_title("Rainfall vs Water Level"); ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("Water level (ft)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Weather records")
    st.dataframe(filtered[["Month","Day","Barangay","Weather","Temperature","Rainfall_mm","Wind","Pressure"]],use_container_width=True)

with tabs[3]:
    agg=filtered.groupby("Barangay").agg(
        Total_Records=("Barangay","size"),
        Mean_Water_Level=("Water_Level_Ft","mean"),
        Max_Water_Level=("Water_Level_Ft","max"),
        Mean_Families=("Families_Affected","mean"),
        Total_Families=("Families_Affected","sum"),
        Total_Damage=("Total_Damage","sum"),
        High_Severity=("Severity",lambda s:(s=="High").sum())
    ).fillna(0)
    features=["Mean_Water_Level","Max_Water_Level","Mean_Families","Total_Families","Total_Damage","Total_Records","High_Severity"]
    X=agg[features].loc[:,agg[features].std(numeric_only=True)>0] if len(agg)>1 else pd.DataFrame(index=agg.index)
    if len(agg)>=3 and X.shape[1]>=1:
        Xs=StandardScaler().fit_transform(X)
        k=min(3,len(agg)-1)
        labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xs)
        agg["Cluster"]=labels
        # Tier 1 means the cluster with the greatest average total damage, not an official warning level.
        damage_order=agg.groupby("Cluster")["Total_Damage"].mean().sort_values(ascending=False).index
        tier={cl:f"Tier {i+1}" for i,cl in enumerate(damage_order)}
        agg["Risk_Tier"]=agg["Cluster"].map(tier)
        sil=silhouette_score(Xs,labels) if len(set(labels))>1 and len(agg)>k else np.nan
        wcss = float(sum(((Xs[labels == cl] - Xs[labels == cl].mean(axis=0)) ** 2).sum() for cl in np.unique(labels)))
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Clusters",k)
        c2.metric("Silhouette score",f"{sil:.3f}" if pd.notna(sil) else "N/A")
        c3.metric("Within-cluster sum of squares",f"{wcss:.2f}")
        c4.metric("Barangays assessed",len(agg))
        fig,ax=plt.subplots(figsize=(8,4)); sns.barplot(data=agg.reset_index(),x="Risk_Tier",y="Total_Damage",ax=ax,errorbar=None,color="#0d9488")
        ax.set_title("Recorded Damage by Relative Risk Tier"); ax.set_ylabel("Total damage (PHP)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(agg.sort_values(["Risk_Tier","Total_Damage"],ascending=[True,False]).round(2),use_container_width=True)
        st.caption("Exploratory grouping based on this dataset only; tiers are relative clusters, not official flood warnings or predictions.")
    else:
        st.info("At least three barangays and usable numeric features are required for clustering.")
        st.dataframe(agg.round(2),use_container_width=True)

with tabs[4]:
    model_data=filtered.dropna(subset=["Water_Level_Ft","Families_Affected","Total_Damage","Severity"]).copy()
    if len(model_data)>=12 and model_data["Severity"].nunique()>=2:
        X=model_data[["Water_Level_Ft","Families_Affected","Total_Damage"]]
        y=model_data["Severity"]
        le=LabelEncoder(); enc=le.fit_transform(y)
        counts=pd.Series(enc).value_counts()
        strat=enc if counts.min()>=2 else None
        X_train,X_test,y_train,y_test=train_test_split(X,enc,test_size=.3,random_state=42,stratify=strat)
        model=DecisionTreeClassifier(max_depth=4,random_state=42,min_samples_leaf=2,class_weight="balanced")
        model.fit(X_train,y_train); pred=model.predict(X_test)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy",f"{accuracy_score(y_test,pred):.3f}")
        c2.metric("Precision",f"{precision_score(y_test,pred,average='macro',zero_division=0):.3f}")
        c3.metric("Recall",f"{recall_score(y_test,pred,average='macro',zero_division=0):.3f}")
        c4.metric("F1 score",f"{f1_score(y_test,pred,average='macro',zero_division=0):.3f}")
        left,right=st.columns(2)
        with left:
            fig,ax=plt.subplots(figsize=(12,6))
            plot_tree(model,feature_names=X.columns.tolist(),class_names=le.classes_.tolist(),filled=True,rounded=True,ax=ax,fontsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with right:
            cm=confusion_matrix(y_test,pred,labels=np.arange(len(le.classes_)))
            fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=le.classes_,yticklabels=le.classes_,ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.warning("Severity labels are derived from the study thresholds for water level, affected families, and total recorded damage. Model results are exploratory and must not be treated as official flood warnings.")
    else:
        st.info(f"There are not enough complete records to train the model. Complete records available: {len(model_data)}. At least 12 records and two severity classes are required.")

with tabs[5]:
    txns=[]
    for _,r in filtered.iterrows():
        items=set()
        if r["Barangay"]!="Unknown": items.add("Barangay="+r["Barangay"])
        if r["Flood_Cause"]!="Unknown": items.add("Cause="+r["Flood_Cause"])
        if r["Severity"]: items.add("Severity="+r["Severity"])
        if r["Season"]!="Unknown": items.add("Season="+r["Season"])
        if r["Month"]!="Unknown": items.add("Month="+r["Month"])
        if items: txns.append(items)
    support=st.slider("Minimum support",0.05,0.80,0.15,0.05)
    confidence=st.slider("Minimum confidence",0.10,1.00,0.50,0.05)
    if len(txns)>=5:
        # Simple Apriori-style itemset counting; cap candidate size for practical performance.
        from itertools import combinations
        counts={}
        n=len(txns)
        for t in txns:
            for size in range(1,min(3,len(t))+1):
                for combo in combinations(sorted(t),size):
                    key=frozenset(combo); counts[key]=counts.get(key,0)+1
        freq={k:v/n for k,v in counts.items() if v/n>=support}
        rules=[]
        for itemset,sup in freq.items():
            if len(itemset)<2: continue
            for item in itemset:
                ant=frozenset([item]); cons=itemset-ant
                ant_sup=freq.get(ant,0); cons_sup=freq.get(cons,0)
                if ant_sup:
                    conf=sup/ant_sup
                    if conf>=confidence:
                        rules.append({"If":", ".join(sorted(ant)),"Then":", ".join(sorted(cons)),"Support":sup,"Confidence":conf,"Lift":conf/cons_sup if cons_sup else np.nan})
        st.caption(f"{len(txns)} transactions · {len(freq)} frequent itemsets · {len(rules)} rules matching the selected thresholds")
        if rules:
            rdf=pd.DataFrame(rules).drop_duplicates().sort_values(["Lift","Confidence"],ascending=False)
            st.dataframe(rdf.head(50).style.format({"Support":"{:.3f}","Confidence":"{:.3f}","Lift":"{:.3f}"}),use_container_width=True)
            fig,ax=plt.subplots(figsize=(8,4)); sc=ax.scatter(rdf["Support"],rdf["Confidence"],c=rdf["Lift"],cmap="viridis",s=70)
            fig.colorbar(sc,ax=ax,label="Lift"); ax.set_xlabel("Support"); ax.set_ylabel("Confidence"); ax.set_title("Association Rules")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else: st.info("No rules meet the selected thresholds. Try lowering the minimum support or confidence.")
    else: st.info("At least five transactions are required for association rule mining.")

with tabs[6]:
    top_b=filtered["Barangay"].value_counts().idxmax() if len(filtered) else "N/A"
    top_month=filtered["Month"].value_counts().idxmax() if len(filtered) else "N/A"
    causes=filtered["Flood_Cause"].replace("Unknown",np.nan).dropna()
    top_cause=causes.mode().iloc[0] if len(causes) else "N/A"
    high_pct=(filtered["Severity"].eq("High").mean()*100) if len(filtered) else 0
    st.markdown(f'<div class="insight-text"><b>Barangay with the highest number of records:</b> {top_b}.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="insight-text"><b>Most frequently recorded month:</b> {top_month}. Note: blank month cells in the workbook are forward-filled where appropriate.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="insight-text"><b>Most common recorded flood cause:</b> {top_cause}.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="insight-text"><b>High-severity records:</b> {high_pct:.1f}% of filtered rows under the study thresholds.</div>',unsafe_allow_html=True)
    st.subheader("Suggested preparedness actions based on historical patterns")
    st.markdown("""- **Prioritize barangays:** Review the K-Means risk profiles together with recorded flood frequency, affected families, water levels, and damage before allocating preparedness resources.
- **Prepare for recurring periods:** Use the monthly and seasonal summaries to plan the timing of monitoring, early coordination, and pre-positioning of essential supplies.
- **Review weather-system patterns:** Use frequent flood causes and high-confidence association rules to inform local monitoring and contingency planning. Association does not prove causation.
- **Strengthen evacuation readiness:** Coordinate barangay-level evacuation planning and communication with MDRRMO/LGU protocols, especially for areas with repeated high-impact records.
- **Validate before implementation:** These are preliminary, data-derived suggestions. They should be reviewed by MDRRMO personnel and checked against current local knowledge and official warning procedures.""")
    st.info("Findings depend on the completeness of the dataset. Blank values do not necessarily mean that no flood event occurred. This dashboard analyzes historical records and is not an official flood forecast. The app-generated recommendations have not been validated by MDRRMO officers.")

st.caption("Flood Analytics Dashboard | Streamlit, pandas, matplotlib, seaborn and scikit-learn")
