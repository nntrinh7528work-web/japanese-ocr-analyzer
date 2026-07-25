"""Youthful & Modern Custom CSS styling for Japanese / English OCR Analyzer."""

import streamlit as st


def inject_custom_css() -> None:
    """Inject modern, vibrant, youthful CSS styles into the Streamlit application."""
    st.markdown(
        """
        <style>
        /* Import Modern Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Main Container Background & Spacing */
        .stApp {
            background-color: #FAFAFF;
        }

        /* Card & Container Polish - Large Radius & Soft Shadows */
        [data-testid="stExpander"] {
            border: 1px solid #E9D5FF;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(124, 58, 237, 0.05);
            margin-bottom: 16px;
            background-color: #FFFFFF;
            overflow: hidden;
            transition: all 0.2s ease-in-out;
        }
        [data-testid="stExpander"]:hover {
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.1);
            border-color: #C084FC;
        }

        /* Metric Box Styling - Soft Vibrant Violet Tint */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #F3F0FF 0%, #E9D5FF 100%);
            border-radius: 14px;
            padding: 14px 18px;
            border: 1px solid #DDD6FE;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.04);
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #4C1D95;
            font-size: 0.85rem;
        }
        [data-testid="stMetricValue"] {
            font-weight: 800;
            color: #6D28D9;
        }

        /* Subheader Accent Line with Soft Gradient */
        .stSubheader {
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #7C3AED 0%, #C084FC 60%, transparent 100%) 1;
            padding-bottom: 8px;
            margin-bottom: 16px;
            color: #1E1B4B;
            font-weight: 700;
        }

        /* File Uploader Dropzone - Youthful Dotted Border */
        [data-testid="stFileUploaderDropzone"] {
            min-height: 140px;
            border: 2px dashed #8B5CF6;
            border-radius: 16px;
            background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
            transition: all 0.25s ease-in-out;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #7C3AED;
            background: #EDE9FE;
            box-shadow: 0 4px 16px rgba(124, 58, 237, 0.12);
        }
        [data-testid="stFileUploaderDropzone"] button {
            min-height: 44px;
            font-weight: 700;
            border-radius: 12px;
        }

        /* Buttons Styling - Vibrant Accent & Micro-animations */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
            color: #FFFFFF;
            font-weight: 700;
            letter-spacing: 0.3px;
            border: none;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6D28D9 0%, #5B21B6 100%);
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.45);
        }

        /* Youthful Dialogue Chat Bubbles */
        .speaker-bubble-a {
            border-left: 5px solid #7C3AED;
            background: linear-gradient(135deg, #F3F0FF 0%, #EDE9FE 100%);
            padding: 14px 18px;
            border-radius: 0 16px 16px 0;
            margin-bottom: 10px;
            color: #1E1B4B;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.05);
        }
        .speaker-bubble-b {
            border-left: 5px solid #EC4899;
            background: linear-gradient(135deg, #FDF2F8 0%, #FCE7F3 100%);
            padding: 14px 18px;
            border-radius: 0 16px 16px 0;
            margin-bottom: 10px;
            color: #831843;
            box-shadow: 0 2px 8px rgba(236, 72, 153, 0.05);
        }

        /* Info & Toast Box Polish */
        .stAlert {
            border-radius: 14px;
            border: 1px solid #DDD6FE;
        }

        /* Tab Navigation Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #F3F0FF;
            color: #7C3AED;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
