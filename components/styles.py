"""Premium Glassmorphism CSS styling (Light & Dark Mode) for Japanese / English OCR Analyzer."""

from __future__ import annotations
import streamlit as st


def build_custom_css(dark_mode: bool = False) -> str:
    """Build the app theme CSS so it can be regression-tested."""
    if dark_mode:
        # ── Dark Mode Palette ──
        bg_app = "#0B0E17"
        bg_gradient = "radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.12) 0%, transparent 50%), radial-gradient(ellipse at 80% 100%, rgba(168,85,247,0.10) 0%, transparent 50%), #0B0E17"
        glass_bg = "rgba(17,20,39,0.72)"
        glass_border = "rgba(148,163,184,0.10)"
        glass_hover_border = "rgba(168,85,247,0.35)"
        glass_shadow = "0 8px 32px rgba(0,0,0,0.35)"
        glass_hover_shadow = "0 12px 40px rgba(139,92,246,0.18)"
        text_primary = "#F1F5F9"
        text_secondary = "#94A3B8"
        text_accent = "#C4B5FD"
        metric_bg = "linear-gradient(135deg, rgba(30,27,75,0.65) 0%, rgba(46,16,101,0.50) 100%)"
        metric_border = "rgba(139,92,246,0.25)"
        metric_label_color = "#A78BFA"
        metric_value_color = "#E9D5FF"
        dropzone_bg = "linear-gradient(135deg, rgba(30,27,75,0.40) 0%, rgba(23,14,56,0.40) 100%)"
        dropzone_border = "rgba(168,85,247,0.45)"
        btn_primary_bg = "linear-gradient(135deg, #7C3AED 0%, #6366F1 100%)"
        btn_primary_hover = "linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%)"
        btn_primary_shadow = "0 4px 20px rgba(124,58,237,0.40)"
        btn_primary_hover_shadow = "0 6px 28px rgba(124,58,237,0.55)"
        bubble_a_bg = "linear-gradient(135deg, rgba(30,27,75,0.60) 0%, rgba(49,46,129,0.40) 100%)"
        bubble_a_border = "#818CF8"
        bubble_a_text = "#E0E7FF"
        bubble_b_bg = "linear-gradient(135deg, rgba(131,24,67,0.35) 0%, rgba(80,7,36,0.35) 100%)"
        bubble_b_border = "#F472B6"
        bubble_b_text = "#FCE7F3"
        tab_active_bg = "rgba(99,102,241,0.18)"
        tab_active_text = "#A5B4FC"
        tab_active_border = "#818CF8"
        sidebar_bg = "rgba(15,18,35,0.85)"
        sidebar_border = "rgba(148,163,184,0.08)"
        divider_color = "rgba(148,163,184,0.10)"
        input_bg = "rgba(30,27,75,0.40)"
        input_border = "rgba(148,163,184,0.15)"
        input_focus_border = "rgba(139,92,246,0.50)"
        scrollbar_thumb = "rgba(139,92,246,0.30)"
        dataframe_header_bg = "rgba(30,27,75,0.60)"
        success_bg = "rgba(16,185,129,0.12)"
        success_border = "rgba(16,185,129,0.30)"
        warning_bg = "rgba(245,158,11,0.12)"
        warning_border = "rgba(245,158,11,0.30)"
        info_bg = "rgba(99,102,241,0.12)"
        info_border = "rgba(99,102,241,0.30)"
        error_bg = "rgba(239,68,68,0.12)"
        error_border = "rgba(239,68,68,0.30)"
    else:
        # ── Light Mode Palette ──
        bg_app = "#F8FAFF"
        bg_gradient = "radial-gradient(ellipse at 20% 0%, rgba(139,92,246,0.06) 0%, transparent 50%), radial-gradient(ellipse at 80% 100%, rgba(99,102,241,0.05) 0%, transparent 50%), #F8FAFF"
        glass_bg = "rgba(255,255,255,0.72)"
        glass_border = "rgba(139,92,246,0.12)"
        glass_hover_border = "rgba(124,58,237,0.30)"
        glass_shadow = "0 4px 24px rgba(139,92,246,0.06)"
        glass_hover_shadow = "0 8px 32px rgba(139,92,246,0.12)"
        text_primary = "#1E1B4B"
        text_secondary = "#6366F1"
        text_accent = "#7C3AED"
        metric_bg = "linear-gradient(135deg, rgba(243,240,255,0.80) 0%, rgba(233,213,255,0.60) 100%)"
        metric_border = "rgba(196,181,253,0.40)"
        metric_label_color = "#6D28D9"
        metric_value_color = "#4C1D95"
        dropzone_bg = "linear-gradient(135deg, rgba(245,243,255,0.80) 0%, rgba(237,233,254,0.70) 100%)"
        dropzone_border = "rgba(139,92,246,0.40)"
        btn_primary_bg = "linear-gradient(135deg, #7C3AED 0%, #6366F1 100%)"
        btn_primary_hover = "linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%)"
        btn_primary_shadow = "0 4px 20px rgba(124,58,237,0.25)"
        btn_primary_hover_shadow = "0 6px 28px rgba(124,58,237,0.40)"
        bubble_a_bg = "linear-gradient(135deg, rgba(243,240,255,0.90) 0%, rgba(224,231,255,0.80) 100%)"
        bubble_a_border = "#6366F1"
        bubble_a_text = "#312E81"
        bubble_b_bg = "linear-gradient(135deg, rgba(253,242,248,0.90) 0%, rgba(252,231,243,0.80) 100%)"
        bubble_b_border = "#EC4899"
        bubble_b_text = "#831843"
        tab_active_bg = "rgba(99,102,241,0.10)"
        tab_active_text = "#4F46E5"
        tab_active_border = "#6366F1"
        sidebar_bg = "rgba(255,255,255,0.60)"
        sidebar_border = "rgba(139,92,246,0.08)"
        divider_color = "rgba(139,92,246,0.10)"
        input_bg = "rgba(243,240,255,0.50)"
        input_border = "rgba(139,92,246,0.15)"
        input_focus_border = "rgba(124,58,237,0.45)"
        scrollbar_thumb = "rgba(139,92,246,0.25)"
        dataframe_header_bg = "rgba(243,240,255,0.80)"
        success_bg = "rgba(16,185,129,0.08)"
        success_border = "rgba(16,185,129,0.25)"
        warning_bg = "rgba(245,158,11,0.08)"
        warning_border = "rgba(245,158,11,0.25)"
        info_bg = "rgba(99,102,241,0.08)"
        info_border = "rgba(99,102,241,0.25)"
        error_bg = "rgba(239,68,68,0.08)"
        error_border = "rgba(239,68,68,0.25)"

    return f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ── CSS Variables Root Override ── */
        :root {{
            --background-color: {bg_app};
            --secondary-background-color: {sidebar_bg};
            --text-color: {text_primary};
        }}

        /* ── Global Reset & Typography ── */
        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background: {bg_gradient} !important;
            color: {text_primary} !important;
        }}

        /* Streamlit's fixed header is outside the main app container. */
        header[data-testid="stHeader"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        .stAppHeader {{
            background: {bg_app} !important;
            color: {text_primary} !important;
        }}

        /* ── Animated accent line at top ── */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #7C3AED, #6366F1, #EC4899, #7C3AED);
            background-size: 300% 100%;
            animation: shimmer 4s ease-in-out infinite;
            z-index: 9999;
        }}
        @keyframes shimmer {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {scrollbar_thumb};
            border-radius: 3px;
        }}

        /* ── Glassmorphism Cards (Expanders & Details) ── */
        [data-testid="stExpander"],
        details {{
            background: {glass_bg} !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {glass_border} !important;
            border-radius: 16px !important;
            box-shadow: {glass_shadow} !important;
            margin-bottom: 12px;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: {text_primary} !important;
        }}
        [data-testid="stExpander"]:hover,
        details:hover {{
            box-shadow: {glass_hover_shadow} !important;
            border-color: {glass_hover_border} !important;
        }}
        [data-testid="stExpanderDetails"],
        [data-testid="stExpander"] > div,
        summary {{
            background: transparent !important;
            color: {text_primary} !important;
        }}
        summary span, [data-testid="stExpander"] summary span {{
            color: {text_primary} !important;
            font-weight: 600;
        }}

        /* ── Container Border Wrappers (Streamlit cards) ── */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {glass_bg} !important;
            border: 1px solid {glass_border} !important;
            border-radius: 16px !important;
            box-shadow: {glass_shadow} !important;
        }}

        /* ── Metrics with glass effect ── */
        [data-testid="stMetric"] {{
            background: {metric_bg} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 14px;
            padding: 16px 20px;
            border: 1px solid {metric_border} !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
            transition: all 0.25s ease;
        }}
        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139,92,246,0.12);
        }}
        [data-testid="stMetricLabel"] {{
            font-weight: 600;
            color: {metric_label_color} !important;
            font-size: 0.8rem;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 800;
            color: {metric_value_color} !important;
            font-size: 1.5rem;
        }}

        /* ── Section Subheaders with gradient underline ── */
        .stSubheader, h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
        }}
        .stSubheader {{
            border-bottom: 2px solid;
            border-image: linear-gradient(90deg, #7C3AED 0%, #6366F1 40%, transparent 100%) 1;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }}

        /* ── Labels & Text ── */
        label, p, li {{
            color: {text_primary} !important;
        }}
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] ol,
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] em {{
            color: {text_primary} !important;
        }}

        /* ── Dividers ── */
        [data-testid="stHorizontalRule"], hr {{
            border-color: {divider_color} !important;
        }}

        /* ── File Uploader & File List Items ── */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"] {{
            min-height: 140px;
            border: 2px dashed {dropzone_border} !important;
            border-radius: 16px !important;
            background: {dropzone_bg} !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: {text_primary} !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {glass_hover_border} !important;
            box-shadow: {glass_hover_shadow};
            transform: scale(1.005);
        }}
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] button {{
            min-height: 40px;
            font-weight: 700;
            border-radius: 12px;
            background: {glass_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {glass_border} !important;
        }}
        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFileData"],
        div[data-testid="stFileUploaderFileContainer"],
        section[data-testid="stFileUploaderDropzone"] + div {{
            background: {glass_bg} !important;
            border: 1px solid {glass_border} !important;
            border-radius: 12px !important;
            color: {text_primary} !important;
        }}
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFileData"] * {{
            color: {text_primary} !important;
        }}

        /* ── Camera Input ── */
        [data-testid="stCameraInput"],
        [data-testid="stCameraInput"] > div {{
            background: {glass_bg} !important;
            border-radius: 16px !important;
            border: 1px solid {glass_border} !important;
            color: {text_primary} !important;
        }}

        /* ── Radio Buttons & Checkboxes / Toggles (Dialogue & Quiz) ── */
        [data-testid="stRadio"],
        [data-testid="stCheckbox"],
        [data-testid="stToggle"],
        div[role="radiogroup"] {{
            color: {text_primary} !important;
        }}
        div[role="radiogroup"] label,
        div[role="radiogroup"] label *,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label * {{
            color: {text_primary} !important;
            background: transparent !important;
        }}
        div[role="radiogroup"] > div {{
            background: transparent !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.2px;
            border: 1px solid {glass_border} !important;
            background: {glass_bg} !important;
            color: {text_primary} !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-color: {glass_hover_border} !important;
        }}
        .stButton > button:active {{
            transform: translateY(0);
        }}
        .stButton > button[kind="primary"] {{
            background: {btn_primary_bg} !important;
            color: #FFFFFF !important;
            font-weight: 700;
            letter-spacing: 0.3px;
            border: none !important;
            box-shadow: {btn_primary_shadow};
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {btn_primary_hover} !important;
            box-shadow: {btn_primary_hover_shadow};
        }}

        /* ── Download buttons ── */
        .stDownloadButton > button {{
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid {glass_border} !important;
            background: {glass_bg} !important;
            color: {text_primary} !important;
        }}
        .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-color: {glass_hover_border} !important;
        }}

        /* ── Dialogue Chat Bubbles ── */
        .speaker-bubble-a {{
            border-left: 4px solid {bubble_a_border} !important;
            background: {bubble_a_bg} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 16px 20px;
            border-radius: 4px 16px 16px 4px;
            margin-bottom: 10px;
            color: {bubble_a_text} !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }}
        .speaker-bubble-a:hover {{
            transform: translateX(4px);
            box-shadow: 0 4px 16px rgba(99,102,241,0.15);
        }}
        .speaker-bubble-b {{
            border-left: 4px solid {bubble_b_border} !important;
            background: {bubble_b_bg} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 16px 20px;
            border-radius: 4px 16px 16px 4px;
            margin-bottom: 10px;
            color: {bubble_b_text} !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            transition: all 0.2s ease;
        }}
        .speaker-bubble-b:hover {{
            transform: translateX(4px);
            box-shadow: 0 4px 16px rgba(236,72,153,0.15);
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {glass_bg} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 14px;
            padding: 4px;
            border: 1px solid {glass_border} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 8px 16px;
            font-weight: 600;
            transition: all 0.2s ease;
            font-size: 0.88rem;
            color: {text_secondary} !important;
            background: transparent !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {tab_active_bg} !important;
            color: {tab_active_text} !important;
            font-weight: 700;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        [data-baseweb="tab-panel"], [data-testid="stTabContent"] {{
            background: transparent !important;
            padding-top: 16px;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background: {sidebar_bg} !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid {sidebar_border} !important;
            color: {text_primary} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            background: transparent !important;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
            border: 1px solid {glass_border} !important;
            box-shadow: none !important;
        }}

        /* ── Text Inputs, Text Areas & BaseWeb Inputs ── */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        div[data-baseweb="textarea"] {{
            background: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
            transition: all 0.2s ease;
            color: {text_primary} !important;
        }}
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color: {text_secondary} !important;
            opacity: 0.85;
        }}

        /* Number input steppers otherwise keep Streamlit's light surface. */
        .stNumberInput button,
        [data-testid="stNumberInput"] button {{
            background: {input_bg} !important;
            color: {text_primary} !important;
            border-color: {input_border} !important;
        }}
        .stNumberInput button svg,
        [data-testid="stNumberInput"] button svg {{
            fill: {text_primary} !important;
            color: {text_primary} !important;
        }}
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        div[data-baseweb="input"]:focus-within {{
            border-color: {input_focus_border} !important;
            box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important;
        }}

        /* ── Select Box & BaseWeb Dropdown Menus ── */
        .stSelectbox > div > div,
        div[data-baseweb="select"] > div {{
            background: {input_bg} !important;
            border: 1px solid {input_border} !important;
            border-radius: 12px !important;
            color: {text_primary} !important;
        }}
        div[data-baseweb="select"] span {{
            color: {text_primary} !important;
        }}
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [data-baseweb="menu"] > div,
        [data-baseweb="menu"] > ul,
        div[role="listbox"],
        ul[role="listbox"] {{
            background: {bg_app} !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {glass_border} !important;
            border-radius: 12px !important;
            box-shadow: {glass_shadow} !important;
        }}
        li[role="option"], [data-baseweb="option"], div[role="option"] {{
            background: {bg_app} !important;
            color: {text_primary} !important;
        }}
        li[role="option"] *, [data-baseweb="option"] *, div[role="option"] * {{
            color: {text_primary} !important;
        }}
        li[role="option"]:hover, [data-baseweb="option"]:hover, div[role="option"]:hover {{
            background: {tab_active_bg} !important;
            color: {tab_active_text} !important;
        }}

        /* ── Alert Boxes Callouts (st.info, st.warning, st.error, st.success) ── */
        [data-testid="stAlert"],
        div[role="alert"] {{
            border-radius: 12px !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            background: {glass_bg} !important;
            border: 1px solid {glass_border} !important;
            color: {text_primary} !important;
        }}
        [data-testid="stAlert"] * {{
            color: {text_primary} !important;
        }}
        div[data-baseweb="notification"] {{
            background: {glass_bg} !important;
            border-radius: 12px !important;
            color: {text_primary} !important;
        }}

        /* ── Progress Bar ── */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #7C3AED, #6366F1);
            border-radius: 6px;
        }}

        /* ── Dataframe & Tables ── */
        [data-testid="stDataFrame"], div[data-testid="stTable"], table {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {glass_border} !important;
            background: {glass_bg} !important;
            color: {text_primary} !important;
        }}
        [data-testid="stDataFrame"] *, table * {{
            color: {text_primary} !important;
            background: transparent !important;
        }}

        /* Dataframe toolbar buttons use a separate BaseWeb surface. */
        [data-testid="stDataFrame"] button,
        [data-testid="stElementToolbar"] button {{
            background: {glass_bg} !important;
            color: {text_primary} !important;
        }}

        /* ── Code blocks ── */
        code, pre {{
            background: {input_bg} !important;
            color: {text_primary} !important;
            border: 1px solid {input_border} !important;
            border-radius: 8px !important;
        }}

        /* ── Captions ── */
        .stCaption, caption {{
            color: {text_secondary} !important;
        }}

        /* ── Toast notifications ── */
        [data-testid="stToast"] {{
            border-radius: 12px;
            background: {glass_bg} !important;
            color: {text_primary} !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {glass_border} !important;
        }}

        /* ── Custom branded header ── */
        .app-header {{
            background: {glass_bg} !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {glass_border} !important;
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: {glass_shadow} !important;
        }}
        .app-header h1 {{
            background: linear-gradient(135deg, #7C3AED, #6366F1, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 900;
            font-size: 2rem;
            margin: 0 0 4px 0;
            letter-spacing: -0.5px;
        }}
        .app-header p {{
            color: {text_secondary} !important;
            margin: 0;
            font-size: 0.92rem;
            font-weight: 400;
        }}

        /* ── Stat badge (for sidebar) ── */
        .stat-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: {metric_bg};
            border: 1px solid {metric_border};
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            color: {metric_label_color} !important;
        }}

        /* ── Streak banner ── */
        .streak-banner {{
            background: linear-gradient(135deg, rgba(245,158,11,0.10) 0%, rgba(249,115,22,0.08) 100%) !important;
            border: 1px solid rgba(245,158,11,0.25) !important;
            border-radius: 14px;
            padding: 14px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 600;
            color: {text_primary} !important;
        }}
        .streak-banner .streak-number {{
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #F59E0B, #F97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Quiz card ── */
        .quiz-card {{
            background: {glass_bg} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid {glass_border} !important;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 12px;
            color: {text_primary} !important;
        }}
        </style>
        """


def inject_custom_css(dark_mode: bool = False) -> None:
    """Inject premium glassmorphism CSS for Light or Dark mode."""
    st.markdown(build_custom_css(dark_mode), unsafe_allow_html=True)


def render_branded_header() -> None:
    """Render a premium branded header with gradient title."""
    st.markdown(
        """
        <div class="app-header">
            <h1>🔍 Japanese / English OCR Analyzer</h1>
            <p>Tải ảnh hoặc PDF → OCR bằng Gemini AI → Phân tích từ vựng, ngữ pháp và mẫu câu chi tiết</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
