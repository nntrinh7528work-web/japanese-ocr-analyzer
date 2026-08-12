"""Client-side synchronized video and bilingual transcript player."""

from __future__ import annotations

import streamlit as st


_PLAYER_HTML = """
<div class="study-player-root">
  <div class="study-media"><div class="youtube-player"></div></div>
  <div class="study-script">
    <div class="script-toolbar"></div>
    <div class="cue-list" aria-live="polite"></div>
  </div>
</div>
"""

_PLAYER_CSS = """
.study-player-root { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(320px,1fr); gap:1rem; width:100%; }
.study-player-root.upload-only { display:block; }
.study-media { position:sticky; top:.75rem; align-self:start; aspect-ratio:16/9; min-height:220px; background:#000; border-radius:.65rem; overflow:hidden; }
.study-media iframe, .youtube-player { width:100%; height:100%; border:0; }
.upload-only .study-media { display:none; }
.study-script { min-width:0; }
.script-toolbar { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-bottom:.55rem; }
.script-toolbar button, .script-toolbar label { border:1px solid var(--st-border-color,#ccc); border-radius:.45rem; padding:.35rem .6rem; background:var(--st-secondary-background-color,#f5f5f5); color:var(--st-text-color,#222); font-size:.9rem; }
.script-toolbar button { cursor:pointer; }
.cue-list { height:520px; overflow-y:auto; overscroll-behavior:contain; border:1px solid var(--st-border-color,#ddd); border-radius:.65rem; padding:.35rem; background:var(--st-background-color,#fff); }
.cue-row { width:100%; box-sizing:border-box; border:0; border-left:4px solid transparent; border-radius:.4rem; background:transparent; color:var(--st-text-color,#222); padding:.6rem .65rem; text-align:left; cursor:pointer; margin:.1rem 0; }
.cue-row:hover { background:color-mix(in srgb, var(--st-primary-color,#ff4b4b) 9%, transparent); }
.cue-row.active { border-left-color:var(--st-primary-color,#ff4b4b); background:color-mix(in srgb, var(--st-primary-color,#ff4b4b) 16%, transparent); }
.cue-meta { display:flex; gap:.45rem; align-items:center; color:var(--st-primary-color,#d33); font-size:.8rem; font-weight:650; }
.cue-source { display:block; font-size:1rem; line-height:1.5; margin-top:.18rem; }
.cue-translation { display:block; font-size:.94rem; line-height:1.45; color:var(--st-text-color,#333); opacity:.84; margin-top:.2rem; }
.cue-pending { font-style:italic; opacity:.58; }
@media (max-width: 760px) {
  .study-player-root { display:block; }
  .study-media { position:sticky; top:0; z-index:5; margin-bottom:.7rem; min-height:190px; }
  .cue-list { height:56vh; min-height:360px; }
  .cue-row { padding:.7rem .55rem; }
}
"""

_PLAYER_JS = r"""
export default function(component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.study-player-root');
  const media = root.querySelector('.study-media');
  const playerNode = root.querySelector('.youtube-player');
  const toolbar = root.querySelector('.script-toolbar');
  const list = root.querySelector('.cue-list');
  const cues = Array.isArray(data?.cues) ? data.cues : [];
  const mode = data?.mode === 'youtube' ? 'youtube' : 'upload';
  root.classList.toggle('upload-only', mode === 'upload');

  let player = null;
  let htmlVideo = null;
  let activeIndex = -1;
  let autoScroll = true;
  let showTranslation = true;
  let disposed = false;

  const formatTime = (value) => {
    const total = Math.max(0, Math.floor(Number(value) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    return hours ? `${hours}:${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}` : `${minutes}:${String(seconds).padStart(2,'0')}`;
  };

  const autoLabel = document.createElement('label');
  const autoBox = document.createElement('input');
  autoBox.type = 'checkbox'; autoBox.checked = true;
  autoBox.onchange = () => { autoScroll = autoBox.checked; };
  autoLabel.append(autoBox, document.createTextNode(' Tự cuộn'));
  const translationLabel = document.createElement('label');
  const translationBox = document.createElement('input');
  translationBox.type = 'checkbox'; translationBox.checked = true;
  translationBox.onchange = () => {
    showTranslation = translationBox.checked;
    list.querySelectorAll('.cue-translation').forEach((node) => node.hidden = !showTranslation);
  };
  translationLabel.append(translationBox, document.createTextNode(' Hiện bản dịch'));
  const returnButton = document.createElement('button');
  returnButton.type = 'button'; returnButton.textContent = 'Về dòng đang phát';
  returnButton.onclick = () => {
    autoScroll = true; autoBox.checked = true;
    list.querySelector('.cue-row.active')?.scrollIntoView({block:'center', behavior:'smooth'});
  };
  toolbar.replaceChildren(autoLabel, translationLabel, returnButton);

  const seek = (seconds) => {
    if (mode === 'youtube' && player?.seekTo) player.seekTo(Number(seconds) || 0, true);
    if (mode === 'upload' && htmlVideo) htmlVideo.currentTime = Number(seconds) || 0;
  };

  const rows = cues.map((cue, index) => {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'cue-row';
    button.onclick = () => seek(cue.start_seconds);
    const meta = document.createElement('span'); meta.className = 'cue-meta';
    const time = document.createElement('span'); time.textContent = formatTime(cue.start_seconds);
    const language = document.createElement('span');
    language.textContent = cue.language === 'japanese' ? 'Tiếng Nhật' : cue.language === 'english' ? 'Tiếng Anh' : '';
    meta.append(time, language);
    const source = document.createElement('span'); source.className = 'cue-source'; source.textContent = cue.source_text || '';
    const translation = document.createElement('span'); translation.className = 'cue-translation';
    translation.textContent = cue.translation_vi || 'Đang chờ bản dịch';
    if (!cue.translation_vi) translation.classList.add('cue-pending');
    button.append(meta, source, translation);
    list.appendChild(button);
    return button;
  });

  list.addEventListener('wheel', () => { autoScroll = false; autoBox.checked = false; }, {passive:true});
  list.addEventListener('touchmove', () => { autoScroll = false; autoBox.checked = false; }, {passive:true});

  const currentTime = () => {
    if (mode === 'youtube' && player?.getCurrentTime) return Number(player.getCurrentTime()) || 0;
    if (mode === 'upload' && htmlVideo) return Number(htmlVideo.currentTime) || 0;
    return 0;
  };
  const updateActive = () => {
    if (!cues.length) return;
    const now = currentTime();
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
    const videos = [...document.querySelectorAll('video')].filter((node) => node.offsetParent !== null);
    htmlVideo = videos.at(-1) || null;
    if (htmlVideo) {
      htmlVideo.setAttribute('playsinline', '');
      htmlVideo.setAttribute('webkit-playsinline', '');
    }
  };

  const loadYouTube = () => {
    if (window.YT?.Player) return Promise.resolve();
    if (window.__ocrYoutubeApiPromise) return window.__ocrYoutubeApiPromise;
    window.__ocrYoutubeApiPromise = new Promise((resolve) => {
      const previous = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => { if (previous) previous(); resolve(); };
      if (!document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) {
        const script = document.createElement('script'); script.src = 'https://www.youtube.com/iframe_api';
        document.head.appendChild(script);
      }
    });
    return window.__ocrYoutubeApiPromise;
  };

  if (mode === 'youtube') {
    loadYouTube().then(() => {
      if (disposed) return;
      player = new window.YT.Player(playerNode, {
        videoId: data.video_id,
        width: '100%', height: '100%',
        playerVars: { playsinline: 1, rel: 0 },
      });
    });
  } else {
    attachUploadVideo();
    setTimeout(attachUploadVideo, 500);
  }
  const timer = setInterval(updateActive, 250);
  return () => {
    disposed = true;
    clearInterval(timer);
    try { player?.destroy?.(); } catch (_) {}
  };
}
"""


_SYNCED_PLAYER = st.components.v2.component(
    "bilingual_video_player",
    html=_PLAYER_HTML,
    css=_PLAYER_CSS,
    js=_PLAYER_JS,
)


def render_youtube_player(video_id: str, cues: list[dict], *, key: str) -> None:
    _SYNCED_PLAYER(
        key=key,
        data={"mode": "youtube", "video_id": video_id, "cues": cues},
        height=650,
    )


def render_upload_transcript(cues: list[dict], *, key: str) -> None:
    _SYNCED_PLAYER(
        key=key,
        data={"mode": "upload", "cues": cues},
        height=590,
    )
