from components.styles import build_custom_css


def test_dark_theme_styles_streamlit_portals_and_header():
    css = build_custom_css(dark_mode=True)

    assert 'header[data-testid="stHeader"]' in css
    assert '[data-baseweb="popover"] > div' in css
    assert 'div[role="option"]' in css
    assert ".stNumberInput button" in css
    assert "#0B0E17" in css


def test_light_and_dark_theme_have_distinct_backgrounds():
    assert "#0B0E17" in build_custom_css(dark_mode=True)
    assert "#F8FAFF" in build_custom_css(dark_mode=False)
