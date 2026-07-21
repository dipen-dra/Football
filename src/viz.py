import sys
# Python 3.9 compatibility patch for newer libraries like mplsoccer
if sys.version_info < (3, 10):
    import dataclasses
    if not hasattr(dataclasses, 'KW_ONLY'):
        dataclasses.KW_ONLY = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from mplsoccer import Pitch, VerticalPitch
from scipy.stats import pearsonr
import warnings
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

# Define color palette matching a premium dark theme
DARK_BG = '#0B0F19'
CARD_BG = '#131926'
TEXT_COLOR = '#F3F4F6'
MUTED_TEXT = '#9CA3AF'
ACCENT_BLUE = '#3B82F6'
ACCENT_GOLD = '#F59E0B'
ACCENT_EMERALD = '#10B981'
ACCENT_RED = '#EF4444'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_BG,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': MUTED_TEXT,
    'ytick.color': MUTED_TEXT,
    'grid.color': '#1E293B',
    'font.family': 'sans-serif'
})

def plot_pressure_heatmap(pressures_df: pd.DataFrame, team_name: str, vertical: bool = False) -> plt.Figure:
    """
    Plot a density heatmap of pressure events for a team on a StatsBomb pitch.
    """
    team_pressures = pressures_df[pressures_df['team'] == team_name].dropna(subset=['x', 'y'])
    
    # Initialize pitch
    try:
        if vertical:
            pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=DARK_BG, line_color='#4a5568', line_zorder=2)
        else:
            pitch = Pitch(pitch_type='statsbomb', pitch_color=DARK_BG, line_color='#4a5568', line_zorder=2)
    except TypeError as e:
        import inspect
        from mplsoccer.soccer.dimensions import FixedDims
        import streamlit as st
        st.error(f"TypeError caught: {e}")
        st.write(f"FixedDims signature: {inspect.signature(FixedDims.__init__)}")
        st.write(f"FixedDims defaults: {FixedDims.__init__.__defaults__}")
        raise e
        
    fig, ax = pitch.draw(figsize=(10, 7) if not vertical else (7, 10))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    if len(team_pressures) > 5:
        # Plot kernel density estimate heatmap
        try:
            # We use kdeplot from seaborn
            kde = sns.kdeplot(
                x=team_pressures['y'] if vertical else team_pressures['x'],
                y=team_pressures['x'] if vertical else team_pressures['y'],
                ax=ax,
                fill=True,
                cmap='hot',
                thresh=0.05,
                alpha=0.6,
                n_levels=20,
                zorder=1
            )
        except Exception as e:
            # Fallback to scatter if KDE fails
            pitch.scatter(team_pressures['x'], team_pressures['y'], ax=ax, color=ACCENT_GOLD, alpha=0.5, s=40, zorder=3)
    else:
        # If too few points, just draw a scatter plot
        pitch.scatter(team_pressures['x'], team_pressures['y'], ax=ax, color=ACCENT_GOLD, alpha=0.7, s=50, zorder=3)
        
    # Draw title
    title_text = f"Pressure Zones: {team_name}\n(Total pressures: {len(team_pressures)})"
    ax.text(
        60 if not vertical else 40, 
        -5 if not vertical else -5, 
        title_text, 
        color=TEXT_COLOR, 
        fontsize=14, 
        ha='center', 
        fontweight='bold'
    )
    
    return fig

def plot_regains_and_shots(pressures_df: pd.DataFrame, team_name: str) -> plt.Figure:
    """
    Plot positions of successful regains and the shots/goals that resulted from them.
    """
    team_data = pressures_df[pressures_df['team'] == team_name].dropna(subset=['x', 'y'])
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=DARK_BG, line_color='#4a5568', line_zorder=2)
    fig, ax = pitch.draw(figsize=(11, 7.5))
    fig.patch.set_facecolor(DARK_BG)
    
    # Successful regains (not necessarily counter-press, just any successful press)
    successful_regains = team_data[team_data['success'] == 1]
    # Regains leading to shots
    shot_regains = team_data[team_data['danger'] == 1]
    # Regains leading to goals
    goal_regains = team_data[team_data['goal'] == 1]
    
    # Scatter plots
    pitch.scatter(
        successful_regains['x'], successful_regains['y'],
        ax=ax, color=ACCENT_EMERALD, alpha=0.6, s=60, edgecolors='#1a202c',
        label='Successful Regain (within 5s)', zorder=3
    )
    
    pitch.scatter(
        shot_regains['x'], shot_regains['y'],
        ax=ax, color=ACCENT_BLUE, alpha=0.9, s=120, marker='^', edgecolors='#1a202c',
        label='Regain Leading to Shot', zorder=4
    )
    
    pitch.scatter(
        goal_regains['x'], goal_regains['y'],
        ax=ax, color=ACCENT_GOLD, alpha=1.0, s=200, marker='*', edgecolors='#1a202c',
        label='Regain Leading to Goal', zorder=5
    )
    
    ax.legend(
        facecolor=CARD_BG, edgecolor='#2d3748', labelcolor=TEXT_COLOR, 
        loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10
    )
    
    title_text = f"Defensive Regains & Attacking Outcome: {team_name}\n({len(successful_regains)} regains, {len(shot_regains)} shots, {len(goal_regains)} goals)"
    ax.text(60, -5, title_text, color=TEXT_COLOR, fontsize=14, ha='center', fontweight='bold')
    
    return fig

def plot_pressing_intensity_timeline(match_events: pd.DataFrame, home_team: str, away_team: str) -> plt.Figure:
    """
    Plot pressing intensity (pressures count in 5-minute rolling windows) over 90 minutes.
    """
    # Sort events by minute
    pressures = match_events[match_events['type'] == 'Pressure'].copy()
    
    # Create minutes list
    minutes_range = list(range(0, 95, 5))
    home_counts = []
    away_counts = []
    
    for m in minutes_range[:-1]:
        start, end = m, m + 5
        home_counts.append(len(pressures[(pressures['team'] == home_team) & (pressures['minute'] >= start) & (pressures['minute'] < end)]))
        away_counts.append(len(pressures[(pressures['team'] == away_team) & (pressures['minute'] >= start) & (pressures['minute'] < end)]))
        
    df_timeline = pd.DataFrame({
        'Interval': [f"{m}-{m+5}'" for m in minutes_range[:-1]],
        'Minute': minutes_range[:-1],
        home_team: home_counts,
        away_team: away_counts
    })
    
    # Smoothen with rolling mean
    df_timeline[f'{home_team}_roll'] = df_timeline[home_team].rolling(window=2, min_periods=1).mean()
    df_timeline[f'{away_team}_roll'] = df_timeline[away_team].rolling(window=2, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    ax.plot(df_timeline['Minute'], df_timeline[f'{home_team}_roll'], color=ACCENT_BLUE, label=home_team, linewidth=2.5)
    ax.plot(df_timeline['Minute'], df_timeline[f'{away_team}_roll'], color=ACCENT_RED, label=away_team, linewidth=2.5)
    
    # Fill under curves
    ax.fill_between(df_timeline['Minute'], df_timeline[f'{home_team}_roll'], color=ACCENT_BLUE, alpha=0.1)
    ax.fill_between(df_timeline['Minute'], df_timeline[f'{away_team}_roll'], color=ACCENT_RED, alpha=0.1)
    
    # Style chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4a5568')
    ax.spines['bottom'].set_color('#4a5568')
    
    ax.set_xlabel('Match Minute', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_ylabel('Pressures (Rolling Avg)', color=TEXT_COLOR, fontsize=11, fontweight='bold')
    ax.set_title('Pressing Intensity Timeline (90 Mins)', color=TEXT_COLOR, fontsize=13, fontweight='bold', pad=15)
    
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3748', labelcolor=TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_scatter(
    matches_df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    x_label: str, 
    y_label: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive Plotly scatter plot with correlation details and regression line.
    """
    df = matches_df.copy()
    
    # Drop NaNs
    df = df.dropna(subset=[x_col, y_col])
    
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    
    # Calculate correlation stats
    r_val, p_val = pearsonr(x, y) if len(x) > 2 else (0.0, 1.0)
    
    # Fits a line
    m, c = np.polyfit(x, y, 1) if len(x) > 2 else (0.0, 0.0)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = m * x_fit + c
    
    fig = go.Figure()
    
    # Scatter points
    if 'opponent' in df.columns and 'match_result' in df.columns:
        hover_text = df.apply(
            lambda r: f"<b>{r['team']}</b> vs {r['opponent']}<br>Result: {r['match_result']}<br>{x_label}: {r[x_col]:.2f}<br>{y_label}: {r[y_col]:.2f}", 
            axis=1
        )
    else:
        hover_text = df.apply(
            lambda r: f"<b>{r['team']}</b><br>{x_label}: {r[x_col]:.2f}<br>{y_label}: {r[y_col]:.2f}", 
            axis=1
        )
    
    if color_col and color_col in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            text=df['team'].apply(lambda t: t.replace(" Women's", "")), # Clean up label text
            textposition='top center',
            textfont=dict(size=9, color='#9CA3AF'),
            marker=dict(
                size=13,
                color=df[color_col],
                colorscale='GnBu', # Premium Green-Blue colorscale
                showscale=True,
                colorbar=dict(
                    title=dict(text=color_col.replace('_', ' ').title(), font=dict(color=TEXT_COLOR, size=10)), 
                    tickcolor=TEXT_COLOR,
                    thickness=15
                ),
                line=dict(width=1.5, color='#0B0F19')
            ),
            hovertext=hover_text,
            hoverinfo='text',
            name='Teams'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            text=df['team'].apply(lambda t: t.replace(" Women's", "")),
            textposition='top center',
            textfont=dict(size=9, color='#9CA3AF'),
            marker=dict(
                size=13,
                color='#10B981', # Cyber emerald accent
                line=dict(width=1.5, color='#0B0F19')
            ),
            hovertext=hover_text,
            hoverinfo='text',
            name='Teams'
        ))
        
    # Regression line
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        line=dict(color='#F59E0B', width=2, dash='dash'), # Amber accent
        name=f"Trendline (r = {r_val:.2f})"
    ))
    
    # Style layout
    fig.update_layout(
        title=dict(
            text=f"Correlation: {x_label} vs {y_label} (r = {r_val:.2f}, p-val = {p_val:.4f})",
            font=dict(color=TEXT_COLOR, size=14, family='Space Grotesk')
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(color=TEXT_COLOR, size=11)),
            gridcolor='#1E293B', 
            zerolinecolor='#1E293B', 
            tickfont=dict(color=MUTED_TEXT, size=10)
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color=TEXT_COLOR, size=11)),
            gridcolor='#1E293B', 
            zerolinecolor='#1E293B', 
            tickfont=dict(color=MUTED_TEXT, size=10)
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)', 
            font=dict(color=TEXT_COLOR, size=10)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig
