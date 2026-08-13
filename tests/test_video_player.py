from components.video_player import _PLAYER_CSS, _PLAYER_HTML, _PLAYER_JS


def test_upload_player_sets_inline_attributes_before_media_attachment():
    assert 'playsinline webkit-playsinline preload="metadata"' in _PLAYER_HTML
    setup = _PLAYER_JS.index("uploadPlayer.setAttribute('playsinline'")
    attachment = _PLAYER_JS.index("const attachUploadVideo")
    assert setup < attachment


def test_upload_player_hides_native_only_after_metadata_and_has_fallback():
    metadata = _PLAYER_JS.index("loadedmetadata")
    hide_native = _PLAYER_JS.index("hiddenNative.style.display = 'none'")
    assert metadata < hide_native
    assert "Player gốc đã được khôi phục" in _PLAYER_JS
    assert "if (hiddenNative) hiddenNative.style.display = ''" in _PLAYER_JS


def test_player_sync_is_local_and_mobile_layout_keeps_script_visible():
    assert "setInterval(updateActive, 250)" in _PLAYER_JS
    assert "fetch(" not in _PLAYER_JS
    assert "XMLHttpRequest" not in _PLAYER_JS
    assert "max-height:38vh" in _PLAYER_CSS
    assert ".cue-list { height:55vh" in _PLAYER_CSS
    assert "requestPictureInPicture" in _PLAYER_JS
    assert "clearInterval(attachTimer)" in _PLAYER_JS
