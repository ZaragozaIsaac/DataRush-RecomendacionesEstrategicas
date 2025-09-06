import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

plt.style.use("bmh")
plt.rcParams["savefig.bbox"] = None
plt.rcParams["savefig.dpi"] = 150

st.set_page_config(page_title="Feriados y viajes", layout="wide")

def render(fig):
    st.pyplot(fig, clear_figure=True, bbox_inches=None)

def load_csv(fname: str) -> pd.DataFrame:
    here = Path(fname)
    alt1 = Path("mnt/data")/fname
    alt2 = Path("/mnt/data")/fname
    for p in (here, alt1, alt2):
        if p.exists():
            return pd.read_csv(p)
    st.error(f"No encontré {fname}. Colócalo junto a la app o en /mnt/data.")
    st.stop()

@st.cache_data
def prepare():
    countries  = load_csv("countries.csv").rename(columns={"alpha_3":"ISO3","name":"country_name"})
    holidays   = load_csv("global_holidays.csv")
    passengers = load_csv("monthly_passengers.csv")

    holidays.columns = [c.strip() for c in holidays.columns]
    holidays["Date"]  = pd.to_datetime(holidays["Date"], errors="coerce")
    holidays["Year"]  = holidays["Date"].dt.year
    holidays["Month"] = holidays["Date"].dt.month

    hol_agg = (holidays.groupby(["ISO3","Year","Month"])
               .agg(total_holidays=("Name","count"))
               .reset_index())

    passengers["Date"] = pd.to_datetime(dict(year=passengers["Year"], month=passengers["Month"], day=1))

    merged = (passengers
              .merge(hol_agg, on=["ISO3","Year","Month"], how="left")
              .merge(countries[["ISO3","country_name"]], on="ISO3", how="left"))

    merged["total_holidays"] = merged["total_holidays"].fillna(0).astype(int)

    sums = (merged.groupby("ISO3")[["Total","Domestic","International"]].sum().sum(axis=1))
    valid_iso = sums[sums > 0].index.tolist()
    merged = merged[merged["ISO3"].isin(valid_iso)].copy()

    base = (merged.groupby(["ISO3","Month"])
            [["Total","Domestic","International","total_holidays"]]
            .mean().reset_index()
            .rename(columns={"Total":"base_Total",
                             "Domestic":"base_Domestic",
                             "International":"base_International",
                             "total_holidays":"base_celebrations"}))

    mb = merged.merge(base, on=["ISO3","Month"], how="left")

    for col, bcol in [("Total","base_Total"),("Domestic","base_Domestic"),("International","base_International")]:
        mb[f"{col}_anomaly_pct"] = (mb[col] - mb[bcol]) / mb[bcol] * 100.0

    mb["celebr_dev"] = mb["total_holidays"] - mb["base_celebrations"]

    valid_countries = countries[countries["ISO3"].isin(valid_iso)].copy()
    iso2name = dict(zip(valid_countries["ISO3"], valid_countries["country_name"]))
    name2iso = {v: k for k, v in iso2name.items()}

    return mb, valid_countries, iso2name, name2iso

mb, countries, iso2name, name2iso = prepare()

def with_country_name(df: pd.DataFrame) -> pd.DataFrame:
    if "country_name" not in df.columns:
        df = df.merge(countries[["ISO3","country_name"]], on="ISO3", how="left")
    return df

st.title("Cómo las celebraciones cambian los viajes")
st.caption("Datos 2010–2019. Pasajeros en miles. Cambios % vs el promedio histórico del país en ese mismo mes.")

page = st.sidebar.radio("Sección", ["KPIs y análisis", "Simulador", "Relato visual"])

st.sidebar.header("Filtros")
years = st.sidebar.slider("Años", int(mb["Year"].min()), int(mb["Year"].max()), (2010, 2019))

all_country_names = sorted(countries["country_name"].dropna().unique().tolist())
sel_country_name = st.sidebar.selectbox("País", all_country_names, index=0)
sel_iso = name2iso.get(sel_country_name)

mask = (mb["Year"].between(*years)) & (mb["ISO3"]==sel_iso)
mb_f = with_country_name(mb.loc[mask].copy())

avg_celebr = float(mb_f["total_holidays"].mean()) if len(mb_f)>0 else 0.0
avg_celebr_int = int(round(avg_celebr))
max_month_celebr = int(mb_f["total_holidays"].max()) if len(mb_f)>0 else 0
allowed_max_thr = max(0, max_month_celebr - avg_celebr_int)

min_thr = st.sidebar.number_input(
    "Umbral de celebraciones extra (para análisis)",
    min_value=0, max_value=allowed_max_thr, value=min(1, allowed_max_thr), step=1,
    help="Se limita a que: promedio redondeado + umbral = máximo mensual histórico del país."
)

def detect_events(df: pd.DataFrame, thr: int):
    evt = df.loc[df["celebr_dev"] >= thr, ["ISO3","Date"]].rename(columns={"Date":"event_month"})
    if len(evt) == 0:
        return None, None, pd.DataFrame({"Domestic_anomaly_pct":[0,0,0],"International_anomaly_pct":[0,0,0]}, index=[-1,0,1]), 0
    def windowize(ev, k):
        w = ev.copy()
        w["Date"] = w["event_month"] + pd.offsets.DateOffset(months=k)
        w["window"] = k
        return w[["ISO3","Date","window"]]
    windows = pd.concat([windowize(evt, k) for k in [-1,0,1]], ignore_index=True)
    panel = windows.merge(
        df[["ISO3","Date","Domestic_anomaly_pct","International_anomaly_pct","base_Domestic","base_International"]],
        on=["ISO3","Date"], how="left"
    ).dropna()
    avg_by_window = (panel.groupby("window")[["Domestic_anomaly_pct","International_anomaly_pct"]]
                     .mean().reindex([-1,0,1]))
    n_events = int(len(evt))
    return windows, panel, avg_by_window, n_events

def kpi_block(panel, avg_by_window, n_events, avg_celebr_int):
    if n_events > 0 and panel is not None and len(panel) > 0:
        pct_int_0 = float(round(avg_by_window.loc[0, "International_anomaly_pct"], 2))
        pct_dom_p1 = float(round(avg_by_window.loc[1, "Domestic_anomaly_pct"], 2))
        base_int_event = panel.loc[panel["window"]==0, "base_International"].mean()
        base_int_event = float(np.nan_to_num(base_int_event, nan=0.0))
        extra_int_total_thousands = float(np.nan_to_num(base_int_event * (pct_int_0/100.0) * n_events, nan=0.0))
    else:
        pct_int_0 = pct_dom_p1 = 0.0
        extra_int_total_thousands = 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Promedio de celebraciones/mes", f"{avg_celebr_int}", help="Media entera de celebraciones por país–mes.")
    k2.metric("Viajes internacionales en el mes de celebración", f"{pct_int_0}% vs normal")
    k3.metric("Viajes dentro del país al mes siguiente", f"{pct_dom_p1}% vs normal")
    k4_value = f"{int(round(extra_int_total_thousands))}" if np.isfinite(extra_int_total_thousands) else "—"
    k4.metric("Personas extra (miles)", k4_value)

def plot_window(series, title):
    x = np.array([-1,0,1]); y = series.values.astype(float)
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", linewidth=3)
    ax.fill_between(x, y, 0, alpha=0.18)
    ax.set_xticks(x); ax.set_xticklabels(["Antes","Durante","Después"])
    ax.set_ylabel("Cambio vs un mes normal (%)")
    ax.set_title(title)
    ax.axhline(0, linestyle="--", linewidth=1)
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.1f}%", ha="center", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    fig.tight_layout()
    render(fig)

if page == "KPIs y análisis":
    st.subheader("KPIs principales")
    windows, panel, avg_by_window, n_events = detect_events(mb_f, min_thr)
    kpi_block(panel, avg_by_window, n_events, avg_celebr_int)

    with st.expander("¿Cómo se eligen los meses y qué es “mes normal”?"):
        st.write(
            "- Mes normal: promedio histórico de pasajeros del país en ese mismo mes.\n"
            "- Mes de celebración: año–mes del país con más celebraciones de lo normal, al menos el umbral.\n"
            "- Incluimos todos los meses que cumplen la condición en el rango.\n"
            "- Antes/Durante/Después: comparamos cada uno contra su propio mes normal y luego promediamos."
        )

    st.subheader("Antes / Durante / Después")
    c1, c2 = st.columns(2)
    with c1:
        plot_window(avg_by_window["Domestic_anomaly_pct"], "Viajes dentro del país")
    with c2:
        plot_window(avg_by_window["International_anomaly_pct"], "Viajes internacionales")

    st.markdown("### Resumen del país")
    if len(mb_f)==0:
        st.info("Sin datos para este filtro.")
    else:
        max_celebr = int(mb_f["total_holidays"].max())
        peaks = mb_f.loc[mb_f["total_holidays"]==max_celebr, ["Date","Month","International_anomaly_pct","Domestic_anomaly_pct"]].copy()
        month_names = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        peaks["Mes"] = peaks["Month"].map(month_names)
        intl_at_peak = float(peaks["International_anomaly_pct"].mean()) if len(peaks)>0 else 0.0
        dom_next = []
        for _, r in peaks.iterrows():
            d_next = (pd.to_datetime(r["Date"]) + pd.offsets.DateOffset(months=1))
            row_next = mb_f.loc[mb_f["Date"]==d_next]
            if not row_next.empty:
                dom_next.append(float(row_next["Domestic_anomaly_pct"].iloc[0]))
        dom_next_mean = float(np.mean(dom_next)) if dom_next else 0.0
        tabla = pd.DataFrame([{
            "País": sel_country_name,
            "Mayor número de festividades en un mes": max_celebr,
            "Mes(es) pico": ", ".join(sorted(peaks["Mes"].unique().tolist())),
            "Cambio internacional en el mes (%) vs normal": round(intl_at_peak,2),
            "Cambio doméstico al mes siguiente (%) vs normal": round(dom_next_mean,2),
        }])
        st.dataframe(tabla, use_container_width=True)

elif page == "Simulador":
    st.subheader("Simulador por país y mes")
    st.caption("Selecciona país y mes para ver volúmenes estimados (promedio histórico).")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_country_name = sel_country_name
        sim_iso = sel_iso
        st.selectbox("País", all_country_names, index=all_country_names.index(sel_country_name), disabled=True)
    with sim_col2:
        sim_month_name = st.selectbox("Mes", ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])
        month_to_num = {m:i+1 for i,m in enumerate(["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"])}
        sim_month = month_to_num[sim_month_name]

    sel_rows = mb[(mb["ISO3"]==sim_iso) & (mb["Month"]==sim_month)].copy()
    base_dom = float(np.nan_to_num(sel_rows["base_Domestic"].iloc[0] if len(sel_rows)>0 else 0.0, nan=0.0))
    base_int = float(np.nan_to_num(sel_rows["base_International"].iloc[0] if len(sel_rows)>0 else 0.0, nan=0.0))

    sim_k1, sim_k2, sim_k3, sim_k4 = st.columns(4)
    sim_k1.metric("País", sim_country_name)
    sim_k2.metric("Mes", sim_month_name)
    sim_k3.metric("Nacionales estimados (miles)", f"{base_dom:,.0f}")
    sim_k4.metric("Internacionales estimados (miles)", f"{base_int:,.0f}")

    st.markdown("### Volumen mensual (miles)")
    base_series = (mb[mb["ISO3"]==sim_iso]
                   .drop_duplicates(["ISO3","Month"])
                   .sort_values("Month")[["Month","base_Domestic","base_International"]])
    x = base_series["Month"].values.astype(int)
    y_dom = base_series["base_Domestic"].values.astype(float)
    y_int = base_series["base_International"].values.astype(float)
    fig, ax = plt.subplots()
    ax.plot(x, y_dom, marker="o", linewidth=3, label="Dentro del país", color="#4C78A8")
    ax.plot(x, y_int, marker="o", linewidth=3, label="Internacional", color="#F58518")
    ax.fill_between(x, y_int, 0, alpha=0.12, color="#F58518")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(["E","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_ylabel("Pasajeros (miles)")
    ax.set_title(f"Volumen mensual — {sim_country_name}")
    ax.legend()
    fig.tight_layout()
    render(fig)

else:
    st.subheader("Relato visual")
    st.caption("Visualizaciones por país; todo % vs el promedio histórico del país en ese mes.")

    windows, panel, avg_by_window, n_events = detect_events(mb_f, min_thr)

    st.markdown("**1) Calendario de celebraciones vs impulso internacional**")
    if len(mb_f)==0:
        st.info("Sin datos para este filtro.")
    else:
        month_names = ["E","F","M","A","M","J","J","A","S","O","N","D"]
        month_stats = (mb_f.groupby("Month")[["total_holidays","International_anomaly_pct"]]
                       .mean().reindex(range(1,13)).fillna(0.0))

        bar_color = "#4C78A8"
        line_color = "#E45756"

        fig, ax1 = plt.subplots()
        y1 = month_stats["total_holidays"].to_numpy(dtype=float)
        ax1.bar(range(1,13), y1, alpha=0.55, color=bar_color, edgecolor="#2F3B52", linewidth=0.8)
        ax1.set_xlabel("Mes"); ax1.set_xticks(range(1,13)); ax1.set_xticklabels(month_names)
        ax1.set_ylabel("Celebraciones (prom.)", color=bar_color)
        ax1.tick_params(axis='y', labelcolor=bar_color)
        ax1.set_ylim(0, max(1.0, y1.max())*1.25)

        ax2 = ax1.twinx()
        y2 = month_stats["International_anomaly_pct"].to_numpy(dtype=float)
        y2 = np.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)
        ax2.plot(range(1,13), y2, marker="o", linewidth=3, color=line_color)
        ax2.fill_between(range(1,13), y2, 0, alpha=0.12, color=line_color)
        y2_min, y2_max = float(np.min(y2)), float(np.max(y2))
        pad = max(1.0, abs(y2_max - y2_min) * 0.2)
        ax2.set_ylim(y2_min - pad, y2_max + pad)
        ax2.set_ylabel("Cambio internacional (%) vs normal", color=line_color)
        ax2.tick_params(axis='y', labelcolor=line_color)

        max_cel_idx = int(month_stats["total_holidays"].idxmax())
        max_int_idx = int(month_stats["International_anomaly_pct"].idxmax())
        ax1.annotate("⬆ Celebraciones pico",
                     xy=(max_cel_idx, month_stats.loc[max_cel_idx,"total_holidays"]),
                     xytext=(max_cel_idx, ax1.get_ylim()[1]*0.92),
                     ha="center", arrowprops=dict(arrowstyle="->", color=bar_color))
        ax2.annotate("⬆ Impulso internacional",
                     xy=(max_int_idx, month_stats.loc[max_int_idx,"International_anomaly_pct"]),
                     xytext=(max_int_idx, ax2.get_ylim()[1]*0.92),
                     ha="center", arrowprops=dict(arrowstyle="->", color=line_color))

        ax1.set_title(f"{sel_country_name}: ¿sube el impulso internacional?")
        fig.tight_layout()
        render(fig)

    st.markdown("**2) Meses pico de celebraciones**")
    if len(mb_f)>0:
        peaks = (mb_f.groupby("Month")["total_holidays"].mean().reset_index())
        peaks = peaks.sort_values("total_holidays", ascending=True).tail(5)

        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=peaks["total_holidays"].min(), vmax=peaks["total_holidays"].max())
        colors_list = [cmap(norm(v)) for v in peaks["total_holidays"]]

        month_names_full = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        labels = [month_names_full[m] for m in peaks["Month"]]
        fig, ax = plt.subplots()
        ax.barh(labels, peaks["total_holidays"], color=colors_list, edgecolor="#2F3B52")
        ax.set_xlabel("Celebraciones (prom.)")
        ax.set_title(f"{sel_country_name}: top meses por celebraciones")
        for i, v in enumerate(peaks["total_holidays"]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=10)
        fig.tight_layout()
        render(fig)

    st.markdown("**3) Tabla de recomendaciones (Impacto – Esfuerzo – KPI propuesto)**")
    recs = pd.DataFrame([
        {"Sector":"Aerolíneas","Impacto":"Alto","Esfuerzo":"Medio","KPI propuesto":"Factor de ocupación, Anticipación de compra","Cuándo activarlo":"Mes con celebraciones ≥ normal + umbral y mes siguiente"},
        {"Sector":"Turismo/DMO","Impacto":"Alto","Esfuerzo":"Medio","KPI propuesto":"Llegadas, Estancia media","Cuándo activarlo":"Meses pico y fines de semana largos"},
        {"Sector":"Hospedaje","Impacto":"Alto","Esfuerzo":"Medio","KPI propuesto":"Ocupación, Tarifa promedio","Cuándo activarlo":"Meses con mayor intensidad y el posterior"},
        {"Sector":"Retail","Impacto":"Medio","Esfuerzo":"Bajo","KPI propuesto":"Afluencia, Ticket promedio","Cuándo activarlo":"Semana del festivo y días previos"},
        {"Sector":"Transporte terrestre","Impacto":"Medio","Esfuerzo":"Bajo","KPI propuesto":"Pasajeros, Ocupación","Cuándo activarlo":"Puentes y retornos (después)"},
        {"Sector":"OTAs/Agencias","Impacto":"Medio","Esfuerzo":"Medio","KPI propuesto":"Conversión, Cancelaciones","Cuándo activarlo":"Antes del festivo y en meses detectados"}
    ])
    st.dataframe(recs, use_container_width=True)
