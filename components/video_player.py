"""Client-side synchronized bilingual video and transcript player."""

from __future__ import annotations

import streamlit as st


_PLAYER_HTML = """
<div class="study-player-root">
  <div class="study-media">
    <div class="youtube-player"></div>
    <video class="upload-player" controls playsinline webkit-playsinline preload="metadata"></video>
    <div class="media-status" role="status"></div>
  </div>
  <div class="study-script">
    <div class="script-toolbar"></div>
    <div class="cue-list" aria-live="polite"></div>
  </div>
</div>
"""

_PLAYER_CSS = """
.study-player-root { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(320px,1fr); gap:1rem; width:100%; }
.study-media { position:sticky; top:.75rem; align-self:start; aspect-ratio:16/9; min-height:220px; background:#000; border-radius:.65rem; overflow:hidden; }
.study-media iframe, .youtube-player, .upload-player { width:100%; height:100%; border:0; display:block; background:#000; }
.upload-player { object-fit:contain; }
.media-status { position:absolute; inset:auto .5rem .5rem; padding:.35rem .5rem; color:#fff; background:rgba(0,0,0,.72); border-radius:.35rem; font-size:.82rem; display:none; }
.study-script { min-width:0; }
.script-toolbar { display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; margin-bottom:.55rem; }
.script-toolbar button, .script-toolbar label, .script-toolbar select { border:1px solid var(--st-border-color,#ccc); border-radius:.45rem; padding:.35rem .55rem; background:var(--st-secondary-background-color,#f5f5f5); color:var(--st-text-color,#222); font-size:.88rem; }
.script-toolbar button { cursor:pointer; }
.cue-list { height:520px; overflow-y:auto; overscroll-behavior:contain; border:1px solid var(--st-border-color,#ddd); border-radius:.65rem; padding:.35rem; background:var(--st-background-color,#fff); }
.cue-row { width:100%; box-sizing:border-box; border:0; border-left:4px solid transparent; border-radius:.4rem; background:transparent; color:var(--st-text-color,#222); padding:.6rem .65rem; text-align:left; cursor:pointer; margin:.1rem 0; }
.cue-row:hover { background:color-mix(in srgb, var(--st-primary-color,#ff4b4b) 9%, transparent); }
.cue-row.active { border-left-color:var(--st-primary-color,#ff4b4b); background:color-mix(in srgb, var(--st-primary-color,#ff4b4b) 16%, transparent); }
.cue-row.needs-review { box-shadow:inset 0 0 0 1px #d97706; }
.cue-meta { display:flex; gap:.45rem; align-items:center; color:var(--st-primary-color,#d33); font-size:.8rem; font-weight:650; }
.cue-source { display:block; font-size:1rem; line-height:1.5; margin-top:.18rem; }
.cue-translation { display:block; font-size:.94rem; line-height:1.45; color:var(--st-text-color,#333); opacity:.84; margin-top:.2rem; }
.cue-pending { font-style:italic; opacity:.58; }
.study-player-root.compact .study-media { aspect-ratio:21/9; min-height:120px; }
@media (max-width: 760px) {
  .study-player-root { display:block; }
  .study-media { position:sticky; top:0; z-index:20; margin-bottom:.7rem; min-height:0; max-height:38vh; aspect-ratio:16/9; }
  .study-media .upload-player, .study-media .youtube-player { max-height:38vh; }
  .cue-list { height:55vh; min-height:350px; }
  .cue-row { padding:.7rem .55rem; }
  .study-player-root.compact .study-media { max-height:22vh; aspect-ratio:21/9; }
}
"""

_PLAYER_JS = r"""
export default function(component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.study-player-root');
  const media = root.querySelector('.study-media');
  const youtubeNode = root.querySelector('.youtube-player');
  const uploadPlayer = root.querySelector('.upload-player');
  const mediaStatus = root.querySelector('.media-status');
  const toolbar = root.querySelector('.script-toolbar');
  const list = root.querySelector('.cue-list');
  const cues = Array.isArray(data?.cues) ? data.cues : [];
  const mode = data?.mode === 'youtube' ? 'youtube' : 'upload';
  const storageKey = `ocr-video-player:${data?.source_id || data?.video_id || 'default'}`;
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) {}

  youtubeNode.hidden = mode !== 'youtube';
  uploadPlayer.hidden = mode !== 'upload';
  uploadPlayer.setAttribute('playsinline', '');
  uploadPlayer.setAttribute('webkit-playsinline', '');
  uploadPlayer.setAttribute('preload', 'metadata');

  let player = null;
  let nativeVideo = null;
  let hiddenNative = null;
  let activeIndex = -1;
  let autoScroll = saved.autoScroll !== false;
  let showTranslation = saved.showTranslation !== false;
  let disposed = false;
  let lastSavedSecond = -1;
  let attachTimer = null;
  let attachTimeout = null;

  const persist = () => {
    const current = mode === 'youtube' && player?.getCurrentTime ? player.getCurrentTime() : uploadPlayer.currentTime;
    const state = { currentTime:Number(current)||0, playbackRate:Number(uploadPlayer.playbackRate)||1, autoScroll, showTranslation };
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) {}
  };
  const formatTime = (value) => {
    const total = Math.max(0, Math.floor(Number(value) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    return hours ? `${hours}:${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}` : `${minutes}:${String(seconds).padStart(2,'0')}`;
  };
  const makeButton = (label, action) => {
    const button = document.createElement('button'); button.type = 'button'; button.textContent = label; button.onclick = action; return button;
  };

  const autoLabel = document.createElement('label');
  const autoBox = document.createElement('input'); autoBox.type = 'checkbox'; autoBox.checked = autoScroll;
  autoBox.onchange = () => { autoScroll = autoBox.checked; persist(); };
  autoLabel.append(autoBox, document.createTextNode(' Tự cuộn'));
  const translationLabel = document.createElement('label');
  const translationBox = document.createElement('input'); translationBox.type = 'checkbox'; translationBox.checked = showTranslation;
  translationBox.onchange = () => {
    showTranslation = translationBox.checked;
    list.querySelectorAll('.cue-translation').forEach((node) => node.hidden = !showTranslation);
    persist();
  };
  translationLabel.append(translationBox, document.createTextNode(' Hiện bản dịch'));
  const speed = document.createElement('select');
  [0.75, 1, 1.25].forEach((value) => { const option = document.createElement('option'); option.value = String(value); option.textContent = `${value}x`; speed.append(option); });
  speed.value = String(saved.playbackRate || 1);
  speed.onchange = () => { if (mode === 'upload') uploadPlayer.playbackRate = Number(speed.value); persist(); };
  const returnButton = makeButton('Về dòng đang phát', () => {
    autoScroll = true; autoBox.checked = true;
    list.querySelector('.cue-row.active')?.scrollIntoView({block:'center', behavior:'smooth'});
  });
  const compactButton = makeButton('Thu nhỏ video', () => {
    root.classList.toggle('compact');
    compactButton.textContent = root.classList.contains('compact') ? 'Mở rộng video' : 'Thu nhỏ video';
  });
  const pipButton = makeButton('Picture-in-Picture', async () => {
    if (mode !== 'upload') return;
    try {
      if (document.pictureInPictureElement) await document.exitPictureInPicture();
      else if (uploadPlayer.requestPictureInPicture) await uploadPlayer.requestPictureInPicture();
      else if (uploadPlayer.webkitSetPresentationMode) uploadPlayer.webkitSetPresentationMode('picture-in-picture');
    } catch (_) {}
  });
  if (mode !== 'upload' || (!document.pictureInPictureEnabled && !uploadPlayer.webkitSupportsPresentationMode)) pipButton.hidden = true;
  toolbar.replaceChildren(autoLabel, translationLabel, speed, returnButton, compactButton, pipButton);

  const seek = (seconds) => {
    if (mode === 'youtube' && player?.seekTo) player.seekTo(Number(seconds) || 0, true);
    if (mode === 'upload') uploadPlayer.currentTime = Number(seconds) || 0;
  };
  const rows = cues.map((cue) => {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'cue-row';
    if (['needs_review','recheck_ready'].includes(cue.verification_status)) button.classList.add('needs-review');
    button.onclick = () => seek(cue.start_seconds);
    const meta = document.createElement('span'); meta.className = 'cue-meta';
    const time = document.createElement('span'); time.textContent = formatTime(cue.start_seconds);
    const language = document.createElement('span'); language.textContent = cue.language === 'japanese' ? 'Tiếng Nhật' : cue.language === 'english' ? 'Tiếng Anh' : '';
    const quality = document.createElement('span'); quality.textContent = cue.confidence === 'low' ? 'Cần kiểm tra' : '';
    meta.append(time, language, quality);
    const source = document.createElement('span'); source.className = 'cue-source'; source.textContent = cue.source_text || '';
    const translation = document.createElement('span'); translation.className = 'cue-translation'; translation.hidden = !showTranslation;
    translation.textContent = cue.translation_vi || 'Đang chờ bản dịch';
    if (!cue.translation_vi) translation.classList.add('cue-pending');
    button.append(meta, source, translation); list.appendChild(button); return button;
  });
  list.addEventListener('wheel', () => { autoScroll = false; autoBox.checked = false; persist(); }, {passive:true});
  list.addEventListener('touchmove', () => { autoScroll = false; autoBox.checked = false; persist(); }, {passive:true});

  const currentTime = () => mode === 'youtube' && player?.getCurrentTime ? Number(player.getCurrentTime()) || 0 : Number(uploadPlayer.currentTime) || 0;
  const updateActive = () => {
    const now = currentTime();
    if (Math.floor(now) !== lastSavedSecond) { lastSavedSecond = Math.floor(now); persist(); }
    let next = -1;
    for (let index = 0; index < cues.length; index += 1) {
      const start = Number(cues[index].start_seconds) || 0;
      const following = index + 1 < cues.length ? Number(cues[index + 1].start_seconds) : Number(cues[index].end_seconds) + 2;
      if (now >= start && now < Math.max(Number(cues[index].end_seconds) || start, following)) next = index;
      if (start > now) break;
    }
    if (next === activeIndex) return;
    if (activeIndex >= 0) rows[activeIndex]?.classList.remove('active');
    activeIndex = next;
    if (activeIndex >= 0) {
      rows[activeIndex]?.classList.add('active');
      if (autoScroll) rows[activeIndex]?.scrollIntoView({block:'center', behavior:'smooth'});
    }
  };

  const attachUploadVideo = () => {
    if (disposed || mode !== 'upload' || uploadPlayer.src) return;
    const containerClass = data?.native_container_key ? `.st-key-${CSS.escape(data.native_container_key)}` : '';
    const scope = containerClass ? document.querySelector(containerClass) : document;
    const videos = [...(scope || document).querySelectorAll('video')].filter((node) => node !== uploadPlayer);
    nativeVideo = videos.find((node) => node.currentSrc || node.src || node.querySelector('source')) || null;
    const source = nativeVideo?.currentSrc || nativeVideo?.src || nativeVideo?.querySelector('source')?.src || '';
    if (!source) return;
    uploadPlayer.src = source;
    uploadPlayer.playbackRate = Number(saved.playbackRate) || 1;
    uploadPlayer.addEventListener('loadedmetadata', () => {
      if (Number(saved.currentTime) > 0 && Number(saved.currentTime) < uploadPlayer.duration) uploadPlayer.currentTime = Number(saved.currentTime);
      hiddenNative = nativeVideo.closest('[data-testid="stVideo"]') || nativeVideo.parentElement;
      if (hiddenNative) hiddenNative.style.display = 'none';
      mediaStatus.style.display = 'none';
    }, {once:true});
    uploadPlayer.addEventListener('error', () => {
      uploadPlayer.removeAttribute('src');
      if (hiddenNative) hiddenNative.style.display = '';
      mediaStatus.textContent = 'Không thể mở player đồng bộ. Player gốc đã được khôi phục.';
      mediaStatus.style.display = 'block';
    }, {once:true});
  };
  const loadYouTube = () => {
    if (window.YT?.Player) return Promise.resolve();
    if (window.__ocrYoutubeApiPromise) return window.__ocrYoutubeApiPromise;
    window.__ocrYoutubeApiPromise = new Promise((resolve) => {
      const previous = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => { if (previous) previous(); resolve(); };
      if (!document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) {
        const script = document.createElement('script'); script.src = 'https://www.youtube.com/iframe_api'; document.head.appendChild(script);
      }
    });
    return window.__ocrYoutubeApiPromise;
  };

  if (mode === 'youtube') {
    loadYouTube().then(() => {
      if (disposed) return;
      player = new window.YT.Player(youtubeNode, { videoId:data.video_id, width:'100%', height:'100%', playerVars:{playsinline:1, rel:0}, events:{onReady:(event) => { if (Number(saved.currentTime)>0) event.target.seekTo(Number(saved.currentTime), true); }} });
    });
  } else {
    attachUploadVideo();
    attachTimer = setInterval(attachUploadVideo, 250);
    attachTimeout = setTimeout(() => {
      clearInterval(attachTimer);
      if (!uploadPlayer.src) { mediaStatus.textContent = 'Không tìm thấy nguồn video cho player đồng bộ.'; mediaStatus.style.display = 'block'; }
    }, 5000);
  }
  const timer = setInterval(updateActive, 250);
  return () => {
    disposed = true; clearInterval(timer); clearInterval(attachTimer); clearTimeout(attachTimeout); persist();
    if (hiddenNative) hiddenNative.style.display = '';
    try { player?.destroy?.(); } catch (_) {}
  };
}
"""


_SYNCED_PLAYER = st.components.v2.component(
    "bilingual_video_player_v2", html=_PLAYER_HTML, css=_PLAYER_CSS, js=_PLAYER_JS,
)


def render_youtube_player(video_id: str, cues: list[dict], *, key: str) -> None:
    _SYNCED_PLAYER(
        key=key,
        data={"mode": "youtube", "video_id": video_id, "source_id": video_id, "cues": cues},
        height=650,
    )


def render_upload_player(cues: list[dict], *, key: str, native_container_key: str, source_id: str) -> None:
    _SYNCED_PLAYER(
        key=key,
        data={
            "mode": "upload", "source_id": source_id,
            "native_container_key": native_container_key, "cues": cues,
        },
        height=650,
    )


def render_upload_transcript(cues: list[dict], *, key: str) -> None:
    """Backward-compatible transcript-only mount for old callers."""
    _SYNCED_PLAYER(
        key=key,
        data={"mode": "upload", "source_id": key, "native_container_key": "", "cues": cues},
        height=590,
    )
