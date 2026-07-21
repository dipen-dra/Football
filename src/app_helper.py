import sys
# Python 3.9 compatibility patch for newer libraries like mplsoccer
if sys.version_info < (3, 10):
    import dataclasses
    if not hasattr(dataclasses, 'KW_ONLY'):
        dataclasses.KW_ONLY = None

import os
import streamlit as st
import pandas as pd
import joblib
import warnings
from src.ingestion import get_competitions, get_matches, load_all_events_for_matches
from src.features import build_features_pipeline
from src.modeling import train_success_model, train_outcome_model

warnings.filterwarnings("ignore")

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Global layout & theme defaults */
    .stApp {
        background-color: #0B0F19 !important;
        color: #F3F4F6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Clean up page padding & width */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 3.5rem !important;
        padding-right: 3.5rem !important;
        max-width: 1200px !important;
    }
    
    /* Hide Streamlit branding and menus selectively */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    /* Hide colored line at the top */
    div[data-testid="stHeaderDecoration"] {
        display: none !important;
    }
    /* Keep header wrapper visible for the sidebar collapse/expand button, but push it to the background */
    header[data-testid="stHeader"] {
        z-index: 99 !important;
    }
    
    /* Premium Page Header styling */
    .dashboard-header {
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 1.5rem;
    }
    .breadcrumbs {
        font-size: 0.85rem;
        color: #10B981;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .page-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -1px;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2;
    }
    .page-subtitle {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* General Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    p, li, label {
        font-family: 'Inter', sans-serif;
        color: #D1D5DB;
    }
    span, div {
        color: #D1D5DB;
    }

    /* Strong visibility for all standard widget labels */
    div[data-testid="stWidgetLabel"] p {
        color: #F3F4F6 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Premium Glassmorphic Card Containers */
    .metric-card {
        background: #131926;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.05);
    }
    
    .metric-title {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #FFFFFF;
        font-size: 2.25rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
        color: #9CA3AF;
    }
    
    .delta-positive { color: #10B981 !important; font-weight: 600; }
    .delta-negative { color: #EF4444 !important; font-weight: 600; }
    
    /* ══════════════════════════════════════════
       SIDEBAR — Premium Redesign
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #070A12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        width: 260px !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebarHeader"] {
        padding: 0 !important;
        min-height: 0 !important;
        height: 0px !important;
        background-color: transparent !important;
    }
    /* Position the collapse button inside the brand header zone */
    [data-testid="stSidebarHeader"] button {
        position: absolute !important;
        top: 24px !important;
        right: 14px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #F3F4F6 !important;
        z-index: 999999 !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }
    [data-testid="stSidebarHeader"] button:hover {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: rgba(16, 185, 129, 0.35) !important;
        color: #10B981 !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding: 0 !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    [data-testid="stSidebarContent"] > div {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Target Streamlit's inner vertical block scroll container */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* ── Hide the auto-generated "app" entry from nav ── */
    [data-testid="stSidebarNav"] li:first-child {
        display: none !important;
    }

    /* ── Hide the auto nav list entirely (we inject our own) ── */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* ── Sidebar Brand / Logo Zone ── */
    .sidebar-brand {
        padding: 1.75rem 1.25rem 1.25rem;
        background: linear-gradient(180deg, #0B1120 0%, #070A12 100%);
        border-bottom: 1px solid rgba(16, 185, 129, 0.12);
        margin-bottom: 0;
    }
    .sidebar-logo-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .sidebar-logo-icon {
        font-size: 1.6rem;
        line-height: 1;
    }
    .sidebar-brand h2 {
        font-size: 1.15rem !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.3px !important;
    }
    .sidebar-brand p {
        font-size: 0.72rem;
        color: #6B7280;
        margin: 0;
        letter-spacing: 0.3px;
    }
    .sidebar-brand-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 2px 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 999px;
        font-size: 0.68rem;
        color: #10B981;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    /* ── Custom Nav Links ── */
    .sidebar-nav {
        padding: 1rem 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-nav-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        padding: 0 0.5rem;
        margin-bottom: 0.4rem;
    }
    .sidebar-nav a {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 0.52rem 0.75rem !important;
        border-radius: 8px !important;
        color: #9CA3AF !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        transition: all 0.15s ease !important;
        margin-bottom: 2px !important;
    }
    .sidebar-nav a:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #E5E7EB !important;
    }
    .sidebar-nav a.active {
        background: rgba(16, 185, 129, 0.12) !important;
        color: #10B981 !important;
        font-weight: 600 !important;
    }
    .sidebar-nav a .nav-icon {
        font-size: 1rem;
        width: 20px;
        text-align: center;
        flex-shrink: 0;
    }

    /* ── Data Controls Section ── */
    .sidebar-section {
        padding: 1rem 1.25rem;
    }
    .sidebar-section-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.6rem;
    }

    /* ── Sidebar widget label text ── */
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #9CA3AF !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }

    /* ── Sidebar selectbox ── */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #0F1624 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
        font-size: 0.875rem !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: rgba(16, 185, 129, 0.4) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: #6B7280 !important;
        fill: #6B7280 !important;
    }

    /* ── Status pill ── */
    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .sidebar-status.ok {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10B981;
    }
    .sidebar-status.warn {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: #F59E0B;
    }
    .sidebar-stats-row {
        display: flex;
        gap: 8px;
        margin-top: 0.75rem;
    }
    .sidebar-stat-chip {
        flex: 1;
        background: #0F1624;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.5rem 0.6rem;
        text-align: center;
    }
    .sidebar-stat-chip .chip-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1;
    }
    .sidebar-stat-chip .chip-label {
        font-size: 0.62rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }

    /* ── Sidebar action button ── */
    [data-testid="stSidebar"] button[kind="secondary"] {
        width: 100% !important;
        background: #0F1624 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #E5E7EB !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 8px 12px !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        border-color: rgba(16, 185, 129, 0.4) !important;
        color: #10B981 !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        width: 100% !important;
        background: #10B981 !important;
        border: none !important;
        color: #000 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        padding: 8px 12px !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: #059669 !important;
        transform: none !important;
    }

    /* ── Sidebar footer ── */
    .sidebar-footer {
        padding: 1rem 1.25rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: auto;
    }
    .sidebar-footer p {
        font-size: 0.68rem;
        color: #374151;
        margin: 0;
        text-align: center;
        line-height: 1.5;
    }
    
    /* Segmented Control / Tabs Styling */
    div[data-baseweb="tab-list"] {
        background-color: #131926 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 4px !important;
        margin-bottom: 1.5rem !important;
    }
    button[data-baseweb="tab"] {
        border-radius: 6px !important;
        background-color: transparent !important;
        color: #9CA3AF !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
    }
    
    /* Custom Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        padding: 2px 4px !important;
        transition: border-color 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10B981 !important;
    }
    
    /* Sliders */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #10B981 !important;
        border: 2px solid #10B981 !important;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.4) !important;
        height: 18px !important;
        width: 18px !important;
    }
    div[data-testid="stSlider"] div[aria-valuemax] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Buttons: Primary & Action styling */
    button[kind="primary"], button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 0.85rem !important;
        width: 100% !important;
    }
    button[kind="primary"] {
        background: #10B981 !important;
        border: none !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    }
    button[kind="primary"]:hover {
        background: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3) !important;
    }
    button[kind="secondary"] {
        background: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    button[kind="secondary"]:hover {
        border-color: #10B981 !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* Styled Info and Alert boxes */
    div[data-testid="stNotification"] {
        background-color: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #F3F4F6 !important;
    }
    
    /* Dataframe and Tables override */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Divider spacing */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Sidebar navigation links ── */
    [data-testid="stSidebarNav"] ul {
        padding: 0 0.5rem !important;
    }
    [data-testid="stSidebarNav"] li a {
        display: block !important;
        padding: 0.55rem 1rem !important;
        border-radius: 8px !important;
        color: #D1D5DB !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        text-decoration: none !important;
        transition: background 0.15s ease, color 0.15s ease !important;
    }
    [data-testid="stSidebarNav"] li a:hover {
        background: rgba(16, 185, 129, 0.08) !important;
        color: #10B981 !important;
    }
    [data-testid="stSidebarNav"] li a[aria-current="page"],
    [data-testid="stSidebarNav"] li a.active {
        background: rgba(16, 185, 129, 0.14) !important;
        color: #10B981 !important;
        font-weight: 600 !important;
    }
    /* Sidebar nav text spans */
    [data-testid="stSidebarNav"] span {
        color: inherit !important;
    }

    /* ── Dark-themed Dataframe / Tables ── */
    /* NOTE: Only style the outer container — Arrow iframe canvas cannot be CSS-injected */
    div[data-testid="stDataFrame"] {
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Native Streamlit Metric widgets ── */
    div[data-testid="metric-container"] {
        background: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        transition: border-color 0.2s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(16, 185, 129, 0.35) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        color: #9CA3AF !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #10B981 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* ── Radio buttons ── */
    div[data-testid="stRadio"] label {
        color: #D1D5DB !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        color: #D1D5DB !important;
    }

    /* ── Checkboxes ── */
    div[data-testid="stCheckbox"] label p {
        color: #D1D5DB !important;
        font-weight: 500 !important;
    }

    /* ── Number inputs ── */
    div[data-testid="stNumberInput"] input {
        background-color: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #F3F4F6 !important;
        padding: 8px 12px !important;
    }

    /* ── Text inputs ── */
    div[data-testid="stTextInput"] input {
        background-color: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #F3F4F6 !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #6B7280 !important;
    }

    /* ── Expander ── */
    div[data-testid="stExpander"] details {
        background: #131926 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
    }
    div[data-testid="stExpander"] summary {
        color: #F3F4F6 !important;
        font-weight: 600 !important;
        padding: 0.75rem 1rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #10B981 !important;
    }

    /* ── Info / Warning / Error alert boxes ── */
    div[data-testid="stAlert"] {
        background-color: #131926 !important;
        border-radius: 10px !important;
        border-left: 3px solid #10B981 !important;
        color: #F3F4F6 !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: #F3F4F6 !important;
    }

    /* ── Columns gap fix ── */
    div[data-testid="column"] {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* ── Pyplot / Matplotlib figure backgrounds ── */
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Plotly chart containers ── */
    div[data-testid="stPlotlyChart"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }

    /* ── Section subheader spacing ── */
    .section-header {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.4rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(16, 185, 129, 0.25);
    }

    /* ── Pill / Badge ── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-green { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-amber { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
    .badge-red   { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-blue  { background: rgba(59,130,246,0.15); color: #3B82F6; border: 1px solid rgba(59,130,246,0.3); }

</style>
"""

def inject_theme_css():
    """Injects custom styling CSS into Streamlit."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

def render_dark_table(df: "pd.DataFrame", caption: str = "") -> None:
    """
    Renders a pandas DataFrame as a dark-themed HTML table matching the app palette.
    Use this instead of st.dataframe() for confusion matrices and small tables.
    """
    import pandas as pd
    css = """
    <style>
    .dark-table { width:100%; border-collapse: collapse; margin: 0.5rem 0 1rem 0; }
    .dark-table th {
        background: #0B0F19; color: #9CA3AF;
        font-size: 0.78rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.6px;
        padding: 10px 14px; text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .dark-table td {
        background: #131926; color: #F3F4F6;
        font-family: 'Space Grotesk', monospace;
        font-size: 0.95rem; font-weight: 500;
        padding: 10px 14px; text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .dark-table tr:hover td { background: #1a2133; }
    .dark-table td.row-label {
        text-align: left; color: #9CA3AF;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    </style>
    """
    # Build header row
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    header_row = f"<tr><th></th>{headers}</tr>"

    # Build data rows
    rows = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows += f"<tr><td class='row-label'>{idx}</td>{cells}</tr>"

    caption_html = f"<caption style='color:#9CA3AF;font-size:0.82rem;font-weight:600;text-align:left;padding:4px 0 8px;text-transform:uppercase;letter-spacing:0.6px;'>{caption}</caption>" if caption else ""
    st.markdown(f"{css}<table class='dark-table'>{caption_html}{header_row}{rows}</table>", unsafe_allow_html=True)

def init_app(page_title: str, subtitle: str = ""):
    """
    Initializes the page layout, styles, and loads/manages session state data.
    """
    st.set_page_config(
        page_title=f"{page_title} | Football Pressing Analytics",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_theme_css()
    
    # ── Read query parameters if available on startup ──
    q_comp = st.query_params.get("comp")
    q_season = st.query_params.get("season")
    if q_comp is not None and q_season is not None:
        try:
            st.session_state['comp_id'] = int(q_comp)
            st.session_state['season_id'] = int(q_season)
        except ValueError:
            pass
            
    active_comp_id = st.session_state.get('comp_id', 72)
    active_season_id = st.session_state.get('season_id', 30)
    
    # Page Header breadcrumbs and title
    comp_name = st.session_state.get('comp_name', '')
    season_name = st.session_state.get('season_name', '')
    if comp_name:
        breadcrumbs = f"Thesis Dashboard &rsaquo; {page_title} &rsaquo; {comp_name} ({season_name})"
    else:
        breadcrumbs = f"Thesis Dashboard &rsaquo; {page_title}"
        
    subtitle_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumbs">{breadcrumbs}</div>
        <div class="page-title">{page_title}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar: Brand / Logo Zone ──
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo-row">
            <span class="sidebar-logo-icon">⚽</span>
            <div>
                <h2>Pressing Analytics</h2>
                <p>BSc Thesis Dashboard</p>
            </div>
        </div>
        <span class="sidebar-brand-badge">v1.0 · StatsBomb Open Data</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Custom Navigation ──
    pages = [
        ("📊", "Overview",               "/"),
        ("🔍", "League Explorer",        "/League_Explorer"),
        ("🛡️", "Team Deep Dive",         "/Team_Deep_Dive"),
        ("⚔️", "Match Detail",           "/Match_Detail"),
        ("🧠", "Pressing Success Model", "/Pressing_Success_Model"),
        ("🏆", "Match Outcome Model",    "/Match_Outcome_Model"),
        ("📚", "Methodology",            "/Methodology"),
    ]
    
    nav_html = '<div class="sidebar-nav"><div class="sidebar-nav-label">Navigation</div>'
    for icon, label, path in pages:
        is_active = (label == page_title)
        active_cls = " active" if is_active else ""
        link_path = f"{path}?comp={active_comp_id}&season={active_season_id}"
        nav_html += f'<a href="{link_path}" target="_self" class="{active_cls.strip()}"><span class="nav-icon">{icon}</span>{label}</a>'
    nav_html += '</div>'
    st.sidebar.markdown(nav_html, unsafe_allow_html=True)

    # ── Sidebar: Data Controls Section ──
    st.sidebar.markdown('<div class="sidebar-section"><div class="sidebar-section-label">Dataset</div></div>', unsafe_allow_html=True)
    
    # Load competitions list
    if 'competitions' not in st.session_state:
        with st.spinner("Fetching available competitions..."):
            try:
                st.session_state['competitions'] = get_competitions()
            except Exception as e:
                st.sidebar.error(f"Failed to connect to StatsBomb API: {e}")
                st.session_state['competitions'] = pd.DataFrame()
                return False
                
    comps_df = st.session_state['competitions']
    if len(comps_df) == 0:
        st.sidebar.warning("No competitions available. Check your internet connection.")
        return False
        
    # Simplify list for selector
    comps_df['comp_display'] = comps_df.apply(
        lambda r: f"{r['competition_name']} ({r['season_name']})", axis=1
    )
    
    # Determine the selectbox index: prioritize session state if it exists, otherwise use defaults
    default_idx = 0
    saved_comp_id = st.session_state.get('comp_id')
    saved_season_id = st.session_state.get('season_id')
    unique_displays = list(comps_df['comp_display'].unique())
    
    if saved_comp_id is not None and saved_season_id is not None:
        matched_idx = comps_df[
            (comps_df['competition_id'] == saved_comp_id) & (comps_df['season_id'] == saved_season_id)
        ].index.tolist()
        if matched_idx:
            display_val = comps_df.loc[matched_idx[0], 'comp_display']
            if display_val in unique_displays:
                default_idx = unique_displays.index(display_val)
    else:
        # Fallback default: Women's World Cup 2019
        wwc_idx = comps_df[
            (comps_df['competition_id'] == 72) & (comps_df['season_id'] == 30)
        ].index.tolist()
        if wwc_idx:
            display_val = comps_df.loc[wwc_idx[0], 'comp_display']
            if display_val in unique_displays:
                default_idx = unique_displays.index(display_val)
        else:
            # Fallback default: World Cup 2018
            wc_idx = comps_df[
                (comps_df['competition_id'] == 43) & (comps_df['season_id'] == 3)
            ].index.tolist()
            if wc_idx:
                display_val = comps_df.loc[wc_idx[0], 'comp_display']
                if display_val in unique_displays:
                    default_idx = unique_displays.index(display_val)
            
    selected_comp_name = st.sidebar.selectbox(
        "Competition & Season",
        comps_df['comp_display'].unique(),
        index=default_idx
    )
    
    selected_comp_row = comps_df[comps_df['comp_display'] == selected_comp_name].iloc[0]
    comp_id = int(selected_comp_row['competition_id'])
    season_id = int(selected_comp_row['season_id'])
    
    # Store selected IDs in session state and URL query parameters
    st.session_state['comp_id'] = comp_id
    st.session_state['season_id'] = season_id
    st.session_state['comp_name'] = selected_comp_row['competition_name']
    st.session_state['season_name'] = selected_comp_row['season_name']
    st.query_params['comp'] = str(comp_id)
    st.query_params['season'] = str(season_id)
    
    # Check if cached data exists
    processed_dir = "data/processed"
    events_path = os.path.join(processed_dir, f"events_features_{comp_id}_{season_id}.parquet")
    matches_path = os.path.join(processed_dir, f"matches_features_{comp_id}_{season_id}.parquet")
    success_model_path = os.path.join("models", f"success_model_{comp_id}_{season_id}.joblib")
    outcome_model_path = os.path.join("models", f"outcome_model_{comp_id}_{season_id}.joblib")
    
    is_cached = os.path.exists(events_path) and os.path.exists(matches_path)
    
    if not is_cached:
        st.sidebar.markdown('<div class="sidebar-status warn">⚠️ Data not cached yet</div>', unsafe_allow_html=True)
        if st.sidebar.button("📥 Download & Build Pipeline", type="primary", use_container_width=True):
            st.session_state['loading_data'] = True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Fetching matches...")
                matches_df = get_matches(comp_id, season_id)
                progress_bar.progress(10)
                
                # Cap the match download count to 25 to avoid long download times/timeouts,
                # while providing enough data to include draws, wins, and losses.
                max_matches = 25
                if len(matches_df) > max_matches:
                    status_text.text(f"Found {len(matches_df)} matches. Sampling {max_matches} for fast cache build...")
                    matches_df = matches_df.sample(max_matches, random_state=42).reset_index(drop=True)
                
                status_text.text(f"Downloading events for {len(matches_df)} matches...")
                
                def update_progress(current, total, msg):
                    frac = 0.1 + (current / total) * 0.6
                    progress_bar.progress(int(frac * 100))
                    status_text.text(msg)
                    
                raw_events = load_all_events_for_matches(matches_df, progress_callback=update_progress)
                
                status_text.text("Extracting PPDA and pressing features...")
                progress_bar.progress(75)
                events_features, matches_features = build_features_pipeline(raw_events, matches_df)
                
                os.makedirs(processed_dir, exist_ok=True)
                events_features.to_parquet(events_path, index=False)
                matches_features.to_parquet(matches_path, index=False)
                
                progress_bar.progress(85)
                
                status_text.text("Training machine learning models...")
                os.makedirs("models", exist_ok=True)
                
                if len(events_features) >= 10 and len(matches_features) >= 10:
                    success_model_data = train_success_model(events_features, models_dir="models")
                    joblib.dump(success_model_data, success_model_path)
                    
                    outcome_model_data = train_outcome_model(matches_features, models_dir="models")
                    joblib.dump(outcome_model_data, outcome_model_path)
                else:
                    st.error("Too few samples to train models. Raw data was saved but models could not be trained.")
                
                progress_bar.progress(100)
                status_text.success("Pipeline built successfully! Reloading...")
                st.session_state['loading_data'] = False
                st.rerun()
                
            except Exception as e:
                st.session_state['loading_data'] = False
                status_text.error(f"Error processing pipeline: {e}")
                import traceback
                st.text(traceback.format_exc())
                return False
    else:
        # Load from cache if needed
        current_cache_key = f"{comp_id}_{season_id}"
        if st.session_state.get('loaded_cache_key') != current_cache_key:
            with st.spinner("Loading data from cache..."):
                try:
                    st.session_state['events_features'] = pd.read_parquet(events_path)
                    st.session_state['matches_features'] = pd.read_parquet(matches_path)
                    
                    if os.path.exists(success_model_path):
                        st.session_state['success_model_data'] = joblib.load(success_model_path)
                    else:
                        st.session_state['success_model_data'] = None
                        
                    if os.path.exists(outcome_model_path):
                        st.session_state['outcome_model_data'] = joblib.load(outcome_model_path)
                    else:
                        st.session_state['outcome_model_data'] = None
                        
                    st.session_state['loaded_cache_key'] = current_cache_key
                    st.session_state['matches_df'] = get_matches(comp_id, season_id)
                except Exception as e:
                    st.error(f"Error loading cached files: {e}")
                    return False
        
        # ── Status + Stats Chips ──
        if st.session_state.get('loaded_cache_key') == f"{comp_id}_{season_id}":
            events_count = len(st.session_state['events_features'])
            matches_count = len(st.session_state['matches_features']) // 2
            st.sidebar.markdown(f"""
            <div class="sidebar-status ok">✓ Data loaded</div>
            <div class="sidebar-stats-row">
                <div class="sidebar-stat-chip">
                    <div class="chip-value">{matches_count}</div>
                    <div class="chip-label">Matches</div>
                </div>
                <div class="sidebar-stat-chip">
                    <div class="chip-value">{events_count:,}</div>
                    <div class="chip-label">Press Events</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="sidebar-status ok">✓ Cached locally</div>', unsafe_allow_html=True)

    # ── Sidebar Footer ──
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        <p>Final-Year Bachelor Thesis<br>Football Pressing Analytics · 2024</p>
    </div>
    """, unsafe_allow_html=True)
        
    return True

def render_segmented_tabs(tab_options: list, labels: dict, query_param_name: str = "tab") -> str:
    """
    Renders a row of premium segmented buttons that act as tabs.
    Uses clean alphanumeric keys for query parameters to avoid encoding issues,
    and maps them to display labels.
    """
    # Get current parameter
    current_val = st.query_params.get(query_param_name, tab_options[0])
    if current_val not in tab_options:
        current_val = tab_options[0]
        
    cols = st.columns(len(tab_options))
    selected_val = current_val
    
    for idx, opt in enumerate(tab_options):
        with cols[idx]:
            is_active = (opt == current_val)
            label = labels.get(opt, opt)
            # Render styled button
            if st.button(
                label, 
                key=f"seg_tab_{query_param_name}_{opt}", 
                type="primary" if is_active else "secondary", 
                use_container_width=True
            ):
                st.query_params[query_param_name] = opt
                selected_val = opt
                st.rerun()
                
    return selected_val
