"""Custom CSS styling for Japanese / English OCR Analyzer Streamlit UI."""

import streamlit as st


def inject_custom_css() -> None:
    """Inject modern custom CSS styles into the Streamlit application."""
    st.markdown(
        """
        <style>
        /* Card & Container Polish */
        [data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            margin-bottom: 12px;
            background-color: rgba(255, 255, 255, 0.02);
        }
        
        /* Metric Box Styling */
        [data-testid="stMetric"] {
            background: rgba(79, 139, 249, 0.08);
            border-radius: 10px;
            padding: 12px 16px;
            border: 1px solid rgba(79, 139, 249, 0.15);
        }
        [data-testid="stMetricLabel"] {
            font-weight: 500;
        }

        /* Subheader Accent Line */
        .stSubheader {
            border-bottom: 2px solid #4f8bf9;
            padding-bottom: 6px;
            margin-bottom: 14px;
        }

        /* File Uploader Dropzone */
        [data-testid="stFileUploaderDropzone"] {
            min-height: 135px;
            border: 2px dashed #4f8bf9;
            border-radius: 12px;
            background: rgba(79, 139, 249, 0.03);
            transition: all 0.2s ease-in-out;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #3b73e3;
            background: rgba(79, 139, 249, 0.07);
        }
        [data-testid="stFileUploaderDropzone"] button {
            min-height: 44px;
            font-weight: 600;
        }

        /* Button Styling */
        .stButton > button[kind="primary"] {
            font-weight: 600;
            letter-spacing: 0.3px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(79, 139, 249, 0.3);
        }

        /* Dialogue Chat Bubbles */
        .speaker-bubble-a {
            border-left: 4px solid #4f8bf9;
            background: rgba(79, 139, 249, 0.06);
            padding: 10px 14px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 8px;
        }
        .speaker-bubble-b {
            border-left: 4px solid #f9844f;
            background: rgba(249, 132, 79, 0.06);
            padding: 10px 14px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
