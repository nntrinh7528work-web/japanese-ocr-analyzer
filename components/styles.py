"""Youthful & Modern Custom CSS styling (Light & Dark Mode) for Japanese / English OCR Analyzer."""

from __future__ import annotations
import streamlit as st


def inject_custom_css(dark_mode: bool = False) -> None:
    """Inject modern CSS styles for Light Mode or Youthful Soft Dark Mode."""
    if dark_mode:
        bg_app = "#0F172A"
        card_bg = "#18182C"
        card_border = "#2E2A5B"
        card_hover_border = "#A855F7"
        text_primary = "#F8FAFC"
        text_muted = "#CBD5E1"
        metric_bg = "linear-gradient(135deg, #1E1B4B 0%, #2E1065 100%)"
        metric_border = "#4C1D95"
        metric_label = "#C084FC"
        metric_value = "#E9D5FF"
        dropzone_bg = "linear-gradient(135deg, #1E1B4B 0%, #170E38 100%)"
        dropzone_border = "#A855F7"
        btn_primary_bg = "linear-gradient(135deg, #9333EA 0%, #7E22CE 100%)"
        btn_primary_hover = "linear-gradient(135deg, #7E22CE 0%, #6B21A8 100%)"
        bubble_a_bg = "linear-gradient(135deg, #2E1065 0%, #1E1B4B 100%)"
        bubble_a_border = "#C084FC"
        bubble_a_text = "#F3F0FF"
        bubble_b_bg = "linear-gradient(135deg, #831843 0%, #500724 100%)"
        bubble_b_border = "#F472B6"
        bubble_b_text = "#FCE7F3"
        tab_active_bg = "#2E1065"
        tab_active_text = "#C084FC"
    else:
        bg_app = "#FAFAFF"
        card_bg = "#FFFFFF"
        card_border = "#E9D5FF"
        card_hover_border = "#C084FC"
        text_primary = "#1E1B4B"
        text_muted = "#4C1D95"
        metric_bg = "linear-gradient(135deg, #F3F0FF 0%, #E9D5FF 100%)"
        metric_border = "#DDD6FE"
        metric_label = "#4C1D95"
        metric_value = "#6D28D9"
        dropzone_bg = "linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%)"
        dropzone_border = "#8B5CF6"
        btn_primary_bg = "linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%)"
        btn_primary_hover = "linear-gradient(135deg, #6D28D9 0%, #5B21B6 100%)"
        bubble_a_bg = "linear-gradient(135deg, #F3F0FF 0%, #EDE9FE 100%)"
        bubble_a_border = "#7C3AED"
        bubble_a_text = "#1E1B4B"
        bubble_b_bg = "linear-gradient(135deg, #FDF2F8 0%, #FCE7F3 100%)"
        bubble_b_border = "#EC4899"
        bubble_b_text = "#831843"
        tab_active_bg = "#F3F0FF"
        tab_active_text = "#7C3AED"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: {bg_app};
            color: {text_primary};
        }}

        [data-testid="stExpander"] {{
            border: 1px solid {card_border};
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
            margin-bottom: 16px;
            background-color: {card_bg};
            overflow: hidden;
            transition: all 0.2s ease-in-out;
        }}
        [data-testid="stExpander"]:hover {{
            box-shadow: 0 6px 20px rgba(168, 85, 247, 0.15);
            border-color: {card_hover_border};
        }}

        [data-testid="stMetric"] {{
            background: {metric_bg};
            border-radius: 14px;
            padding: 14px 18px;
            border: 1px solid {metric_border};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            font-weight: 600;
            color: {metric_label};
            font-size: 0.85rem;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 800;
            color: {metric_value};
        }}

        .stSubheader {{
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #A855F7 0%, #C084FC 60%, transparent 100%) 1;
            padding-bottom: 8px;
            margin-bottom: 16px;
            color: {text_primary};
            font-weight: 700;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            min-height: 140px;
            border: 2px dashed {dropzone_border};
            border-radius: 16px;
            background: {dropzone_bg};
            transition: all 0.25s ease-in-out;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {card_hover_border};
            box-shadow: 0 4px 16px rgba(168, 85, 247, 0.2);
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            min-height: 44px;
            font-weight: 700;
            border-radius: 12px;
        }}

        .stButton > button {{
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
        }}
        .stButton > button[kind="primary"] {{
            background: {btn_primary_bg};
            color: #FFFFFF;
            font-weight: 700;
            letter-spacing: 0.3px;
            border: none;
            box-shadow: 0 4px 14px rgba(147, 51, 234, 0.35);
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {btn_primary_hover};
            box-shadow: 0 6px 18px rgba(147, 51, 234, 0.5);
        }}

        .speaker-bubble-a {{
            border-left: 5px solid {bubble_a_border};
            background: {bubble_a_bg};
            padding: 14px 18px;
            border-radius: 0 16px 16px 0;
            margin-bottom: 10px;
            color: {bubble_a_text};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        .speaker-bubble-b {{
            border-left: 5px solid {bubble_b_border};
            background: {bubble_b_bg};
            padding: 14px 18px;
            border-radius: 0 16px 16px 0;
            margin-bottom: 10px;
            color: {bubble_b_text};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 12px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {tab_active_bg};
            color: {tab_active_text};
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
