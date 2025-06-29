import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
import base64
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score

# --- PAGE CONFIG & CSS ---
st.set_page_config(page_title="Foreign Arrivals Dashboard", page_icon="🌐", layout="wide")
st.markdown("""
    <style>
        html, body, .stApp {font-family: 'Poppins', 'Inter', 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #f7fafc 0%, #dbeafe 100%);}
        .big-header {font-size: 2.6em; font-weight: 900; background: linear-gradient(90deg,#155e75,#0ea5e9 80%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 8px; letter-spacing: 0.5px;}
        .kpi-card {background: #fff; border-radius: 20px; box-shadow: 0 8px 30px rgba(30,64,175,0.11), 0 1.5px 5px rgba(30,41,59,0.12); padding: 22px 0 16px 0; text-align: center; margin-bottom: 12px; transition: box-shadow 0.2s; border: 1.3px solid #e0e7ef;}
        .kpi-label {font-size: 1.05em; color: #6b7280; font-weight: 500; margin-bottom: 6px;}
        .kpi-val {font-size: 2.25em; font-weight: 900; color: #1e293b; margin-bottom: 3px; letter-spacing: 0.01em;}
        .kpi-sub {color: #16a34a; font-size: 1.12em; font-weight: 600;}
        .section-title {font-size: 1.25em; font-weight: 700; margin: 32px 0 16px 0; letter-spacing: .01em; border-left: 4px solid #0ea5e9; padding-left: 13px; color: #075985;}
        .desc {font-size: 1.14em; color: #475569;}
        .dashboard-divider {border-top: 1.7px solid #e2e8f0; margin: 26px 0;}
        .executive-summary {background: linear-gradient(90deg, #e0f2fe 50%, #f0fdf4 100%); border-left: 6px solid #0ea5e9; padding: 19px 22px 13px 22px; border-radius: 16px; margin-bottom: 20px; font-size: 1.02em; color: #334155; font-weight: 500; box-shadow: 0 1.5px 9px rgba(14,165,233,.05);}
        .footer {color:#64748b; text-align:right; margin-top: 36px; font-size:1.01em;}
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            color: #f1f5f9;
            min-width: 265px !important;
            padding-top: 28px;
            padding-bottom: 24px;
            border-right: 1.5px solid #232633;
        }
        .profile-block {display: flex; align-items: center; gap: 16px; margin-bottom: 24px; margin-top: 6px;}
        .profile-block img {width: 48px; height: 48px; border-radius: 50%; object-fit: cover; border: 2.5px solid #0ea5e9;}
        .profile-block .details .name { font-weight: 600; font-size: 1.15em; color: #f1f5f9; }
        .profile-block .details .email { color: #cbd5e1; font-size: 0.97em; }
        .sidebar-divider { border-bottom: 1px solid #475569; margin: 24px 0 18px 0;}
        .stDataFrame {background: #f8fafc !important; border-radius: 14px !important; border: 1.1px solid #e0e7ef !important;}
        thead tr th {background-color: #e0e7ef !important; color: #334155 !important;}
        tbody tr td {background-color: #fff !important; color: #334155 !important;}
        section[data-testid="stSidebar"] button {
            background: #0ea5e9 !important;    /* Bright blue background */
            color: #fff !important;            /* White text */
            border-radius: 8px !important;
            border: 1.5px solid #0ea5e9 !important;
            margin-bottom: 10px !important;
            font-weight: 600 !important;
            font-size: 1.07em !important;
            box-shadow: 0 2px 8px rgba(14, 165, 233, 0.10);
            opacity: 1 !important;             /* Always fully opaque */
            transition: background 0.15s, color 0.15s;
        }
        section[data-testid="stSidebar"] button:disabled {
            background: #e0e7ef !important;    /* Lighter background when disabled */
            color: #b6bcc7 !important;         /* Faded text when disabled */
            opacity: 1 !important;             /* Still fully opaque, just faded color */
            cursor: not-allowed !important;
        }
        section[data-testid="stSidebar"] button:hover,
        section[data-testid="stSidebar"] button:active {
            background: #2563eb !important;    /* Even deeper blue on hover/click */
            color: #fff !important;
            border-color: #2563eb !important;
        }
        /* FIX: Always show multiselect labels clearly in sidebar */
        section[data-testid="stSidebar"] label {
            color: #cbd5e1 !important;      /* Brighter text for dark sidebar */
            font-weight: 700 !important;    /* Bolder label */
            font-size: 1.07em !important;   /* Slightly bigger */
            margin-bottom: 7px !important;
            display: block !important;
            letter-spacing: .02em;
            padding-left: 1px;
        }
        section[data-testid="stSidebar"] .stMultiSelect {
            margin-bottom: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)


# ========== DATA LOAD ==========
df = pd.read_csv('final_cleaned_dataset.csv')
df['country'] = df['country'].astype(str).str.strip().str.upper()
df['date'] = pd.to_datetime(df['date'])

@st.cache_data
def load_country_centroids():
    centroids = pd.read_csv('countries_codes_and_coordinates.csv')
    centroids.columns = [col.strip() for col in centroids.columns]
    centroids['Alpha-3 code'] = centroids['Alpha-3 code'].astype(str).str.replace('"','').str.strip().str.upper()
    centroids['Latitude (average)'] = pd.to_numeric(
        centroids['Latitude (average)'].astype(str).str.replace('"','').str.strip(), errors='coerce')
    centroids['Longitude (average)'] = pd.to_numeric(
        centroids['Longitude (average)'].astype(str).str.replace('"','').str.strip(), errors='coerce')
    return centroids[['Alpha-3 code', 'Latitude (average)', 'Longitude (average)']]

country_centroids = load_country_centroids()

# ========== SESSION STATE INIT ==========
all_years = [str(y) for y in sorted(df['date'].dt.year.unique())]
all_ports = sorted(df['poe'].unique())
all_countries = sorted(df['country'].unique())

if 'years' not in st.session_state:
    st.session_state.years = all_years.copy()
if 'ports' not in st.session_state:
    st.session_state.ports = all_ports.copy()
if 'countries' not in st.session_state:
    st.session_state.countries = all_countries.copy()

# ========== SIDEBAR ==========
with st.sidebar:
    # Profile Block
    try:
        with open("profile.png", "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
            <div class="profile-block">
                <img src="data:image/png;base64,{encoded}">
                <div class="details">
                    <span class="name">Syusyi</span>
                    <span class="email">nursyasya.aina03@gmail.com</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
            <div class="profile-block">
                <div class="details">
                    <span class="name">Syusyi</span>
                    <span class="email">nursyasya.aina03@gmail.com</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("**Data Filters**", unsafe_allow_html=True)
   
    def clear_years():
        st.session_state.years = []

    def clear_ports():
        st.session_state.ports = []

    def clear_countries():
        st.session_state.countries = []

    coly1, coly2 = st.columns([1, 1])
    with coly1:
        st.button("Select All Years", on_click=lambda: st.session_state.update({"years": all_years}))
        st.button("Select All Ports", on_click=lambda: st.session_state.update({"ports": all_ports}))
        st.button("Select All Countries", on_click=lambda: st.session_state.update({"countries": all_countries}))
    with coly2:
        st.button("Clear All Years", on_click=clear_years)
        st.button("Clear All Ports", on_click=clear_ports)
        st.button("Clear All Countries", on_click=clear_countries)

    # Use key=... for widget value
    st.multiselect("Year(s)", all_years, key="years")
    st.multiselect("Port(s) of Entry", all_ports, key="ports")
    st.multiselect("Country/Countries", all_countries, key="countries")

    # Download filtered CSV
    csv = df.copy()
    if st.session_state.years:
        csv = csv[csv['date'].dt.year.astype(str).isin(st.session_state.years)]
    if st.session_state.ports:
        csv = csv[csv['poe'].isin(st.session_state.ports)]
    if st.session_state.countries:
        csv = csv[csv['country'].isin(st.session_state.countries)]
    csv_bytes = csv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered CSV",
        data=csv_bytes,
        file_name="filtered_arrivals.csv",
        mime="text/csv"
    )
    st.markdown(
        '<div style="color:#cbd5e1;font-size:0.98em;">'
        'Use the filters to customize your analysis.<br>'
        'Download the data for your own research.'
        '</div>', unsafe_allow_html=True
    )

# ========== FILTER DATA ==========
if not st.session_state.years or not st.session_state.ports or not st.session_state.countries:
    filtered_df = df.iloc[0:0]
else:
    filtered_df = df[
        (df['date'].dt.year.astype(str).isin(st.session_state.years)) &
        (df['poe'].isin(st.session_state.ports)) &
        (df['country'].isin(st.session_state.countries))
    ]
show_empty = filtered_df.empty



# ========== TABS ==========
tabs = st.tabs([
    "Overview", 
    "Trends", 
    "Demographic", 
    "Geographics", 
    "Forecast", 
    "Data Table"
])

# ========== TAB 1: Overview ==========
with tabs[0]:
    st.markdown('<div class="big-header">FOREIGN ARRIVAL TO MALAYSIA DASHBOARD (2020-2023) </div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="executive-summary" style="margin-bottom:12px;">
            <b>Executive Summary</b>
            <ul style="margin:0 0 0 18px;">
                <li>Arrivals dropped sharply in 2020-2021 due to COVID-19 border closures.</li>
                <li>Strong recovery in 2022-2023 as Malaysia reopened, led by Singaporean visitors and Johor POEs.</li>
                <li>Forecasts predict continued growth, with southern entry points staying busiest.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    # =============================
    # KPI CARDS: Consistent & Aligned
    # =============================

    # CSS to force same height for all cards
    card_style = """
    <style>
    .kpi-card {
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    # Split into 4 even columns for cards
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            '<div class="kpi-card"><span style="font-size:2.2em;">🧳</span><div class="kpi-label">Total Arrivals</div><div class="kpi-val">{:,}</div></div>'
            .format(int(filtered_df['arrivals'].sum())), unsafe_allow_html=True)

    with k2:
        st.markdown(
            '<div class="kpi-card"><span style="font-size:2.2em;">🌏</span><div class="kpi-label">Countries</div><div class="kpi-val">{}</div></div>'
            .format(filtered_df['country'].nunique()), unsafe_allow_html=True)

    with k3:
        st.markdown(
            '<div class="kpi-card"><span style="font-size:2.2em;">🛬</span><div class="kpi-label">Ports of Entry</div><div class="kpi-val">{}</div></div>'
            .format(filtered_df['poe'].nunique()), unsafe_allow_html=True)

    with k4:
        busiest_poe = filtered_df.groupby('poe')['arrivals'].sum().idxmax() if not filtered_df.empty else '-'
        busiest_poe_count = int(filtered_df.groupby('poe')['arrivals'].sum().max()) if not filtered_df.empty else 0
        st.markdown(
            f'''<div class="kpi-card">
                    <span style="font-size:2.2em;">🚩</span>
                    <div class="kpi-label">Busiest POE</div>
                    <div class="kpi-val" style="font-size:1.18em;line-height:1.19;">{busiest_poe}</div>
                    <div class="kpi-sub" style="margin-top:2px; color: #16a34a; font-size: 1.1em;">({busiest_poe_count:,})</div>
                </div>
            ''', unsafe_allow_html=True)

    # =============================
    # Top 5 Countries Bar Chart
    # =============================

    st.markdown("### Top 5 Countries by Arrivals")

    top_countries = filtered_df.groupby('country')['arrivals'].sum().nlargest(5).reset_index()
    total_top = top_countries['arrivals'].sum()
    top_countries['percentage'] = (top_countries['arrivals'] / total_top * 100).round(2)

    flag_dict = {
        'SINGAPORE': '🇸🇬',
        'INDONESIA': '🇮🇩',
        'THAILAND': '🇹🇭',
        'CHINA': '🇨🇳',
        'BRUNEI DARUSSALAM': '🇧🇳',
    }
    top_countries['flag'] = top_countries['country'].map(flag_dict).fillna('')

    bar_labels = [
        f"{v:,.0f} ({p}%)" for v, p in zip(top_countries['arrivals'], top_countries['percentage'])
    ]

    # More vibrant palette
    colorful_palette = px.colors.qualitative.Vivid

    # After calculating top_countries and before fig_bar creation
    max_val = top_countries['arrivals'].max()   

    fig_bar = go.Figure(go.Bar(
        x=top_countries['arrivals'],
        y=[f"{flag} {country}" for flag, country in zip(top_countries['flag'], top_countries['country'])],
        orientation='h',
        marker=dict(color=['#5db0ff', '#ff7675', '#a29bfe', '#ffe066', '#77dd77']), 
        text=[f"{v:,} ({p}%)" for v, p in zip(top_countries['arrivals'], top_countries['percentage'])],
        textposition='outside',  
        hovertemplate='<b>%{y}</b><br>Arrivals: %{x:,}<br>Percent: %{text}<extra></extra>',
    ))

    fig_bar.update_layout(
        xaxis_title="Number of Arrivals",
        yaxis_title="Country",
        height=370,
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        margin=dict(l=70, r=40, t=40, b=40),
        font=dict(size=16, family='Poppins, Inter, Arial, sans-serif'),
    )

    fig_bar.update_xaxes(range=[0, max_val * 1.15]) 

    st.plotly_chart(fig_bar, use_container_width=True)

# ========== TAB 2: Trends ==========
with tabs[1]:
    st.markdown('<div class="section-title">Monthly Arrival Trends</div>', unsafe_allow_html=True)
    if show_empty:
        st.info("No data available for this filter selection. Please select a filter to visualize trends.")
    else:
        min_date = filtered_df['date'].min()
        max_date = filtered_df['date'].max()
        date_range = st.date_input("Select date range", (min_date, max_date))
        trend_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(date_range[0])) &
                               (filtered_df['date'] <= pd.to_datetime(date_range[1]))]
        monthly = trend_df.groupby(trend_df['date'].dt.to_period('M'))['arrivals'].sum().reset_index()
        monthly['date'] = monthly['date'].astype(str)
        monthly['7d_ma'] = monthly['arrivals'].rolling(7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['arrivals'], mode='lines', name='Monthly Arrivals', line=dict(width=3, color='#2563eb')))
        fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['7d_ma'], mode='lines', name='7-day Moving Avg', line=dict(width=2, color='#f59e42', dash='dot')))
        fig.update_layout(hovermode='x', height=340)
        st.plotly_chart(fig, use_container_width=True)

        monthly['pct_change'] = monthly['arrivals'].pct_change()
        if not monthly.empty and monthly['pct_change'].notnull().any():
            min_row = monthly.iloc[monthly['pct_change'].idxmin()]
            max_row = monthly.iloc[monthly['pct_change'].idxmax()]
            col1, col2 = st.columns(2)
            with col1: st.metric("Sharpest Drop", f"{min_row['date']} ({int(min_row['arrivals']):,})", f"{min_row['pct_change']*100:.1f}%")
            with col2: st.metric("Biggest Jump", f"{max_row['date']} ({int(max_row['arrivals']):,})", f"+{max_row['pct_change']*100:.1f}%")
        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Foreign Arrivals by Region (Yearly Proportion)</div>', unsafe_allow_html=True)
        # (100% stacked bar or facet bar as discussed earlier)
        if not filtered_df.empty and 'region' in filtered_df.columns:
            df_region = filtered_df.copy()
            df_region['year'] = df_region['date'].dt.year
            arrivals_by_region_year = (
                df_region.groupby(['region', 'year'])['arrivals']
                .sum()
                .reset_index()
            )
            arrivals_by_region_year['region'] = arrivals_by_region_year['region'].fillna("Unknown")
            pivot = arrivals_by_region_year.pivot(index='region', columns='year', values='arrivals').fillna(0)
            region_totals = pivot.sum(axis=1)
            for c in pivot.columns:
                pivot[c] = pivot[c] / region_totals
            pivot = pivot.reset_index().melt(id_vars='region', var_name='year', value_name='proportion')
            # For clarity, remove regions with no data
            pivot = pivot[pivot['proportion'] > 0]
            fig = px.bar(
                pivot,
                x='proportion', y='region', color='year', orientation='h',
                barmode='stack',
                labels={'proportion': 'Proportion of Total Arrivals'},
                text=pivot['proportion'].apply(lambda x: f"{x:.0%}")
            )
            fig.update_traces(textposition='inside', insidetextanchor='middle')
            fig.update_layout(
                xaxis_tickformat='.0%',
                xaxis_title="Yearly Proportion of Arrivals (per Region)",
                yaxis_title="Region",
                legend_title="Year",
                height=440,
                bargap=0.25,
                margin=dict(l=80, r=10, t=30, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No region data available for this selection.")
        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Foreign Arrivals by Type of Point of Entry (by Year)</div>', unsafe_allow_html=True)

        if not filtered_df.empty and 'type' in filtered_df.columns and 'date' in filtered_df.columns:
            arrivals_by_type_year = (
                filtered_df
                .assign(year=filtered_df['date'].dt.year)
                .groupby(['year', 'type'])['arrivals']
                .sum()
                .reset_index()
            )

            # --- Get max value for y-axis padding ---
            y_max = arrivals_by_type_year['arrivals'].max()

            fig_type = px.bar(
                arrivals_by_type_year,
                x='year', y='arrivals', color='type',
                barmode='group', 
                labels={'arrivals': 'Number of Arrivals', 'year': 'Year', 'type': 'Type of Point of Entry'},
                text=arrivals_by_type_year['arrivals'].apply(lambda x: f"{x/1_000_000:.2f}M"),
                color_discrete_sequence=['#fdba74', '#6ee7b7', '#93c5fd'],
                height=410
            )
            fig_type.update_traces(textposition='outside', textfont=dict(size=15))
            fig_type.update_layout(
                title="Foreign Arrivals by Type of Point of Entry (2020-2023)",
                xaxis_title="Year",
                yaxis_title="Number of Arrivals",
                legend_title="Point of Entry Type",
                font=dict(size=16, family='Poppins, Inter, Arial, sans-serif'),
                margin=dict(l=60, r=40, t=60, b=40), 
                plot_bgcolor="#fff",
                paper_bgcolor="#fff"
            )
            fig_type.update_yaxes(range=[0, y_max * 1.18])  
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("No entry type data available for the selected filters.")

        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)


    if not filtered_df.empty and 'date' in filtered_df.columns:

        st.markdown('<div class="section-title">Compare Monthly Trends by Country</div>', unsafe_allow_html=True)
        country_options = filtered_df['country'].unique().tolist()
        selected_trend_countries = st.multiselect(
            "Pick up to 3 countries for trend comparison", country_options,
            default=country_options[:2] if len(country_options) >= 2 else country_options, max_selections=3, key="trends_countries"
        )
        if selected_trend_countries:
            comp_df = filtered_df[filtered_df['country'].isin(selected_trend_countries)].copy()
            comp_df['YearMonth'] = comp_df['date'].dt.to_period('M').astype(str)
            line_data = comp_df.groupby(['YearMonth', 'country'])['arrivals'].sum().reset_index()
            fig_multi = px.line(line_data, x='YearMonth', y='arrivals', color='country',
                                labels={'YearMonth': 'Month', 'arrivals': 'Arrivals', 'country': 'Country'},
                                template='plotly_white')
            fig_multi.update_layout(height=340, hovermode='x')
            st.plotly_chart(fig_multi, use_container_width=True)
        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Animated Arrivals by Country Over Time</div>', unsafe_allow_html=True)
        anim_df = trend_df.copy()
        anim_df['YearMonth'] = anim_df['date'].dt.to_period('M').astype(str)
        anim_top = anim_df.groupby(['YearMonth', 'country'])['arrivals'].sum().reset_index()
        if not anim_top.empty:
            top10 = anim_top.groupby('country')['arrivals'].sum().nlargest(10).index
            anim_top = anim_top[anim_top['country'].isin(top10)]
            fig_anim = px.bar(
                anim_top, x='country', y='arrivals', color='country', animation_frame='YearMonth',
                range_y=[0, anim_top['arrivals'].max()*1.15],
                title='Arrivals by Country Over Time'
            )
            fig_anim.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            st.info("No data for animated chart.")

# ========== TAB 3: Demographic ==========
with tabs[2]:
    st.markdown('<div class="section-title">Arrivals by Country</div>', unsafe_allow_html=True)
    demog_df = filtered_df.copy()
    # --- Top N Countries Bar Chart ---
    top_n = st.slider("Show Top N Countries", min_value=5, max_value=30, value=15)
    arrivals_by_country = demog_df.groupby('country')['arrivals'].sum().sort_values(ascending=False)
    top_countries = arrivals_by_country.head(top_n)
    other_sum = arrivals_by_country[top_n:].sum()
    if other_sum > 0:
        top_countries['Other'] = other_sum
    # Make a DataFrame for color mapping
    top_countries_df = pd.DataFrame({
        'country': top_countries.index,
        'arrivals': top_countries.values
    })
    fig_country = px.bar(
        top_countries_df,
        x='country',
        y='arrivals',
        labels={'country':'Country','arrivals':'Total Arrivals'},
        color='country',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_country.update_layout(
        xaxis_tickangle=-35,
        showlegend=False,
        title=f"Arrivals by Country (Top {top_n} + Other)"
    )
    st.plotly_chart(fig_country, use_container_width=True)
    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

    # --- Gender Pie Chart ---
    st.markdown('<div class="section-title">Arrivals by Gender )</div>', unsafe_allow_html=True)
    total_male = demog_df['arrivals_male'].sum() if 'arrivals_male' in demog_df else 0
    total_female = demog_df['arrivals_female'].sum() if 'arrivals_female' in demog_df else 0
    fig_gender = px.pie(
        names=['Male', 'Female'],
        values=[total_male, total_female],
        color_discrete_sequence=['#60a5fa', '#f472b6'],
        hole=0.5
    )
    fig_gender.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_gender, use_container_width=True)
    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

    # --- Gender Over Time ---
    st.markdown('<div class="section-title">Arrivals by Gender Over Time</div>', unsafe_allow_html=True)
    if 'arrivals_male' in demog_df and 'arrivals_female' in demog_df:
        gender_time = demog_df.groupby(['date'])[['arrivals_male','arrivals_female']].sum().reset_index()
        fig_gender_time = go.Figure()
        fig_gender_time.add_trace(go.Scatter(
            x=gender_time['date'], y=gender_time['arrivals_male'],
            mode='lines', name='Male', line=dict(color='#60a5fa')
        ))
        fig_gender_time.add_trace(go.Scatter(
            x=gender_time['date'], y=gender_time['arrivals_female'],
            mode='lines', name='Female', line=dict(color='#f472b6')
        ))
        fig_gender_time.update_layout(height=340, xaxis_title='Date', yaxis_title='Arrivals', hovermode='x')
        st.plotly_chart(fig_gender_time, use_container_width=True)
    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

# ========== TAB 4: Geographics ==========
with tabs[3]:
    st.markdown('<div class="section-title">Arrivals by Port (Geographics)</div>', unsafe_allow_html=True)
    if show_empty:
        st.info("No data available for the selected filters.")
    else:
        geo_df = filtered_df.copy()
        # --- Top N Ports Bar Chart ---
        top_n_ports = st.slider("Show Top N Ports", min_value=5, max_value=30, value=15, key="ports_slider")
        arrivals_by_port = geo_df.groupby('poe')['arrivals'].sum().sort_values(ascending=False)
        top_ports = arrivals_by_port.head(top_n_ports)
        other_sum_port = arrivals_by_port[top_n_ports:].sum()
        if other_sum_port > 0:
            top_ports['Other'] = other_sum_port
        top_ports_df = pd.DataFrame({
            'poe': top_ports.index,
            'arrivals': top_ports.values
        })
        fig_port = px.bar(
            top_ports_df,
            x='poe',
            y='arrivals',
            labels={'poe':'Port of Entry','arrivals':'Total Arrivals'},
            color='poe',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_port.update_layout(
            xaxis_tickangle=-35,
            showlegend=False,
            title=f"Arrivals by Port (Top {top_n_ports} + Other)"
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # --- Top 5 Ports by POE Type (Side by Side) ---
        st.markdown('<div class="section-title">Top 5 Ports by POE Type (Side by Side)</div>', unsafe_allow_html=True)
        if not geo_df.empty and 'type' in geo_df.columns:
            unique_types = geo_df['type'].dropna().unique()
            cols = st.columns(len(unique_types))
            colors = ['#60a5fa', '#f472b6', '#f59e42', '#a3e635', '#c084fc']
            for idx, poe_type in enumerate(unique_types):
                with cols[idx]:
                    st.markdown(f"<b>{poe_type} Entry</b>", unsafe_allow_html=True)
                    ports = (geo_df[geo_df['type'] == poe_type]
                            .groupby('poe')['arrivals'].sum()
                            .sort_values(ascending=False)
                            .head(5)
                            .reset_index())
                    fig = go.Figure(go.Bar(
                        x=ports['arrivals'],
                        y=ports['poe'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=ports['arrivals'].apply(lambda x: f"{x:,}"),
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f"Top 5 {poe_type} Ports",
                        xaxis_title="Arrivals",
                        yaxis_title=None,
                        height=325,
                        margin=dict(l=5, r=5, t=32, b=5),
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for POE type and port.")

        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)

        # --- Top 5 States Bar Chart ---
        st.markdown('<div class="section-title">Top 5 States for Foreign Arrivals</div>', unsafe_allow_html=True)
        if not geo_df.empty and 'state' in geo_df.columns:
            arrivals_by_state = (
                geo_df.groupby('state')['arrivals']
                .sum()
                .reset_index()
                .sort_values('arrivals', ascending=False)
                .head(5)
            )
            total = arrivals_by_state['arrivals'].sum()
            arrivals_by_state['Percentage'] = arrivals_by_state['arrivals'] / total * 100
            arrivals_by_state['Label'] = (
                arrivals_by_state['arrivals'].apply(lambda x: f"{x/1_000_000:.2f}M")
                + " (" + arrivals_by_state['Percentage'].round(2).astype(str) + "%)"
            )
            fig_state = go.Figure()
            fig_state.add_trace(go.Bar(
                x=arrivals_by_state['arrivals'],
                y=arrivals_by_state['state'],
                orientation='h',
                marker=dict(color=['#ef4444', '#f59e1e', '#a855f7', '#3b82f6', '#22c55e']),
                text=arrivals_by_state['Label'],
                textposition='auto',
                insidetextanchor='middle'
            ))
            fig_state.update_layout(
                xaxis_title="Count",
                yaxis_title="State",
                title="Top 5 States for Foreign Arrivals",
                height=380,
                template='plotly_white'
            )
            st.plotly_chart(fig_state, use_container_width=True)
        else:
            st.info("No state data available for the selected filters.")

        st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)


        # --- Map ---
        st.markdown('<div class="section-title">Country-Port Flows (Map)</div>', unsafe_allow_html=True)
        n_top = st.slider("Number of Flows", 5, 20, 10, key="geo_map_slider")
        flows = geo_df
        if not flows.empty and 'lat' in flows and 'long' in flows:
            top_flows = flows.groupby(['country', 'poe', 'lat', 'long'])['arrivals'].sum().reset_index()
            top_flows = top_flows.sort_values('arrivals', ascending=False).head(n_top)
            top_flows = pd.merge(top_flows, country_centroids, how='left', left_on='country', right_on='Alpha-3 code')
            fig = px.scatter_geo()
            for idx, (_, row) in enumerate(top_flows.iterrows()):
                origin_lat = row['Latitude (average)']
                origin_lon = row['Longitude (average)']
                dest_lat = row['lat']
                dest_lon = row['long']
                color = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
                fig.add_trace(go.Scattergeo(
                    lon = [origin_lon, dest_lon], lat = [origin_lat, dest_lat],
                    mode = 'lines+markers', line = dict(width = 3, color = color),
                    marker = dict(size = 6, color = color),
                    name = f"{row['country']} → {row['poe']} ({row['arrivals']:.0f})",
                    hoverinfo = 'text',
                    text = f"{row['country']} → {row['poe']}<br>Arrivals: {row['arrivals']:.0f}"
                ))
            fig.update_layout(
                geo=dict(
                    scope='asia', showland=True, landcolor='rgb(230, 232, 236)', showcountries=True,
                    center=dict(lat=3.5, lon=102), lonaxis=dict(range=[80, 125]), lataxis=dict(range=[-10, 30])
                ),
                margin={"r":0,"t":16,"l":0,"b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for map.")

# ========== TAB 5: Forecast ==========
with tabs[4]:
    st.markdown('<div class="section-title">Forecast: Next N Months (Linear Regression)</div>', unsafe_allow_html=True)
    if show_empty:
        st.info("No data available for this filter selection.")
    else:
        if len(st.session_state.ports) != 1:
            st.warning("Please select exactly one Port of Entry for forecasting.")
        else:
            n_months = st.slider("How many months to forecast?", min_value=3, max_value=36, value=18)
            forecast_df = filtered_df.copy()
            forecast_df['YearMonth'] = forecast_df['date'].dt.to_period('M').astype(str)
            monthly_arrivals = forecast_df.groupby('YearMonth')['arrivals'].sum().reset_index()
            monthly_arrivals['month_num'] = np.arange(len(monthly_arrivals))
            if len(monthly_arrivals) < 12:
                st.warning("Not enough data to generate a reliable forecast (need at least 12 months).")
            else:
                # --- Split data: 80% train, 20% test ---
                split_index = int(len(monthly_arrivals) * 0.8)
                train = monthly_arrivals.iloc[:split_index]
                test = monthly_arrivals.iloc[split_index:]

                X_train = train['month_num'].values.reshape(-1, 1)
                y_train = train['arrivals'].values
                X_test = test['month_num'].values.reshape(-1, 1)
                y_test = test['arrivals'].values

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                # --- Evaluate model performance ---
                mae = mean_absolute_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)

                # --- Display performance results ---
                st.markdown("#### Model Evaluation on Past Data")
                st.markdown(f"- <b>Mean Absolute Error (MAE):</b> <span style='color:#1e40af'>{mae:,.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"- <b>R² Score:</b> <span style='color:#1e40af'>{r2:.3f}</span>", unsafe_allow_html=True)
                perf_df = pd.DataFrame({
                    'Metric': ['Mean Absolute Error (MAE)', 'R² Score'],
                    'Value': [f"{mae:,.2f}", f"{r2:.3f}"]
                })
                st.table(perf_df)

                # --- Fit on ALL data for future forecast ---
                X = monthly_arrivals['month_num'].values.reshape(-1, 1)
                y = monthly_arrivals['arrivals'].values
                model.fit(X, y)

                # --- Forecast future ---
                future_months = np.arange(X.max() + 1, X.max() + n_months + 1).reshape(-1, 1)
                y_pred = model.predict(future_months)
                last_date = pd.to_datetime(monthly_arrivals['YearMonth'].iloc[-1] + '-01')
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, n_months+1)]
                future_year_month = [d.strftime("%Y-%m") for d in future_dates]

                # --- Combine for plotting ---
                full_dates = monthly_arrivals['YearMonth'].tolist() + future_year_month
                full_arrivals = list(y) + list(y_pred)
                port_label = st.session_state.ports[0]
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=monthly_arrivals['YearMonth'], y=monthly_arrivals['arrivals'],
                    mode='lines+markers', name='Historical', line=dict(color='#2563eb', width=3)
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_year_month, y=y_pred,
                    mode='lines+markers', name=f'Forecast (Next {n_months} Months)', line=dict(color='#22c55e', dash='dash', width=3)
                ))
                fig_forecast.update_layout(
                    xaxis_title='Year-Month', yaxis_title='Arrivals',
                    title=f'Historical & Predicted Arrivals ({port_label})',
                    height=400, hovermode='x'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                pred_table = pd.DataFrame({'Month': future_year_month, 'Predicted Arrivals': y_pred.astype(int)})
                st.markdown("**Forecast Table:**")
                st.dataframe(pred_table, use_container_width=True)

# ========== TAB 6: Data Table ==========
with tabs[5]:
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

# ========== FOOTER ==========
st.markdown('<hr style="border:none; border-top:1px solid #eee; margin-top:32px;">', unsafe_allow_html=True)
st.markdown('<div class="footer">© 2024 SKIH3033 InfoVis Project &nbsp;|&nbsp; Dashboard by <b>Nursyasya Aina</b></div>', unsafe_allow_html=True)

