"""Streamlit view for structured daily conversation practice."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from modules.cost_estimator import estimate_cost
from modules.dialogue_exporter import export_dialogue_to_docx, export_dialogue_to_json, export_dialogue_to_text
from modules.dialogue_generator import generate_dialogue, generate_variation, suggest_topics, suggest_vocab_grammar
from modules.dialogue_history import (
    get_due_sm2_cards,
    get_practice_history,
    get_streak_days,
    load_dialogue_session,
    save_dialogue_session,
    update_quiz_score,
    update_sm2_card,
)
from modules.dialogue_quiz import generate_pure_quiz, normalize_quiz_answer
from modules.tts_engine import get_audio_cache_key, get_last_tts_error, generate_full_dialogue_audio, text_to_speech


def _is_english(language: str) -> bool:
    return "english" in str(language or "").lower() or "tiếng anh" in str(language or "").lower()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "-", str(value or "hoi-thoai")).strip(" .")
    return cleaned[:80] or "hoi-thoai"


def _role_map(result: dict[str, Any]) -> dict[str, dict[str, str]]:
    rows = result.get("roles") if isinstance(result.get("roles"), list) else []
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("id", "")).upper() in {"A", "B"}:
            output[str(row["id"]).upper()] = row
    return output


def _record_quiz_answer(question_id: str, correct: bool, session_id: str) -> None:
    results = dict(st.session_state.get("dialogue_quiz_results", {}))
    results[question_id] = bool(correct)
    st.session_state.dialogue_quiz_results = results
    score = round(100 * sum(results.values()) / len(results)) if results else 0
    history_id = st.session_state.get("dialogue_history_id")
    if history_id:
        update_quiz_score(history_id, score, session_id=session_id)


def _render_srs(session_id: str) -> None:
    streak = get_streak_days(session_id=session_id)
    due_cards = get_due_sm2_cards(session_id=session_id)
    col1, col2 = st.columns(2)
    col1.metric("Chuỗi ngày luyện", f"{streak} ngày")
    col2.metric("Mục cần ôn hôm nay", len(due_cards))
    if due_cards:
        with st.expander("Ôn tập ngắt quãng", expanded=False):
            for card in due_cards[:5]:
                st.markdown(f"**{card['word']}**")
                if card.get("reading"):
                    st.caption(card["reading"])
                st.caption(card["meaning"])
                ratings = st.columns(4)
                for label, rating, column in (("Quên", 0, ratings[0]), ("Khó", 3, ratings[1]), ("Tốt", 4, ratings[2]), ("Rất dễ", 5, ratings[3])):
                    if column.button(label, key=f"sm2_{rating}_{card['id']}", use_container_width=True):
                        update_sm2_card(card["id"], rating, session_id=session_id)
                        st.rerun()


def _render_turn(turn: dict[str, Any], index: int, roles: dict[str, dict[str, str]], *, show_hiragana: bool, show_translation: bool, enable_tts: bool, tts_lang: str, tts_slow: bool) -> None:
    speaker = str(turn.get("speaker", "A")).upper()
    role = roles.get(speaker, {})
    role_name = role.get("name") or speaker
    with st.chat_message("user" if speaker == "A" else "assistant"):
        st.markdown(f"**{role_name}** · {turn.get('speech_intent') or 'lượt nói'}")
        st.write(turn.get("text", ""))
        if show_hiragana and turn.get("text_hira"):
            st.caption(turn["text_hira"])
        if enable_tts:
            key = "tts_turn_" + get_audio_cache_key(turn.get("text", ""), tts_lang, tts_slow, speaker)
            if st.button("Nghe câu này", key=f"tts_{turn.get('id', index)}"):
                audio = text_to_speech(turn.get("text", ""), lang=tts_lang, slow=tts_slow, speaker=speaker)
                if audio:
                    st.session_state[key] = audio
                else:
                    st.warning(get_last_tts_error() or "Không thể tạo audio cho câu này.")
            if st.session_state.get(key):
                st.audio(st.session_state[key], format="audio/mp3")
        reveal_key = f"reveal_translation_{turn.get('id', index)}"
        if show_translation or st.session_state.get(reveal_key):
            st.caption("Tiếng Việt: " + (turn.get("text_vi") or "Chưa có bản dịch."))
        elif st.button("Hiện bản dịch", key=f"reveal_{turn.get('id', index)}"):
            st.session_state[reveal_key] = True
            st.rerun()
        with st.expander("Ghi chú cách nói", expanded=False):
            if turn.get("register"):
                st.write("Phong cách: " + turn["register"])
            if turn.get("naturalness_note"):
                st.write("Điểm tự nhiên: " + turn["naturalness_note"])
            if turn.get("alternative_expression"):
                st.write("Cách nói khác: " + turn["alternative_expression"])
            if turn.get("highlights"):
                st.write("Mục tiêu học: " + ", ".join(turn["highlights"]))


def _render_quiz(quiz: dict[str, list[dict[str, Any]]], is_english: bool, session_id: str) -> None:
    tabs = st.tabs(["Điền từ", "Trắc nghiệm nghĩa", "Sắp xếp câu", "Dịch câu"])
    with tabs[0]:
        for question in quiz.get("cloze", []):
            st.write(question["sentence"])
            answer = st.radio("Chọn từ", question["options"], key=f"ui_{question['id']}", index=None)
            if st.button("Kiểm tra", key=f"check_{question['id']}"):
                correct = answer == question["target_word"]
                _record_quiz_answer(question["id"], correct, session_id)
                st.success("Đúng.") if correct else st.error("Đáp án: " + question["target_word"])
    with tabs[1]:
        for question in quiz.get("mcq", []):
            st.write(question["question"])
            answer = st.radio("Chọn nghĩa", question["options"], key=f"ui_{question['id']}", index=None)
            if st.button("Kiểm tra", key=f"check_{question['id']}"):
                correct = answer == question["correct_answer"]
                _record_quiz_answer(question["id"], correct, session_id)
                st.success("Đúng.") if correct else st.error("Đáp án: " + question["correct_answer"])
    with tabs[2]:
        for question in quiz.get("reorder", []):
            st.caption("Nghĩa: " + question.get("text_vi", ""))
            st.write(" | ".join(question["shuffled_tokens"]))
            answer = st.text_input("Nhập câu hoàn chỉnh", key=f"ui_{question['id']}")
            if st.button("Kiểm tra", key=f"check_{question['id']}"):
                correct = normalize_quiz_answer(answer, question.get("language", "")) == normalize_quiz_answer(question["original"], question.get("language", ""))
                _record_quiz_answer(question["id"], correct, session_id)
                st.success("Đúng.") if correct else st.info("Câu mẫu: " + question["original"])
    with tabs[3]:
        language_name = "Anh" if is_english else "Nhật"
        for question in quiz.get("translate", []):
            st.write(f"Dịch sang tiếng {language_name}: {question['prompt_vi']}")
            st.text_input("Câu trả lời của bạn", key=f"ui_{question['id']}")
            if st.button("Hiện câu mẫu", key=f"check_{question['id']}"):
                st.info(question["correct_text"])
                if question.get("text_hira"):
                    st.caption(question["text_hira"])


def _render_history(session_id: str) -> None:
    st.divider()
    st.subheader("Lịch sử luyện tập")
    rows = get_practice_history(limit=10, session_id=session_id)
    if not rows:
        st.caption("Các hội thoại đã tạo sẽ xuất hiện ở đây.")
        return
    labels = {f"#{row['id']} · {row['topic']} · {row['created_at'][:16]}": row["id"] for row in rows}
    selected = st.selectbox("Mở bài đã lưu", list(labels), key="dialogue_history_picker")
    if st.button("Mở bài", key="open_dialogue_history"):
        result = load_dialogue_session(labels[selected], session_id=session_id)
        if result:
            st.session_state.dialogue_result = result
            st.session_state.dialogue_quiz = generate_pure_quiz(result)
            st.session_state.dialogue_history_id = labels[selected]
            st.session_state.dialogue_quiz_results = {}
            st.rerun()


def render_dialogue_tab(
    session_id: str = "default",
    model_name: str | None = None,
    reasoning_effort: str = "standard",
    billing_tier: str = "free",
    usd_to_jpy: float = 155.0,
) -> None:
    st.subheader("Luyện hội thoại hằng ngày")
    _render_srs(session_id)

    with st.expander("1. Thiết lập bài luyện", expanded=not bool(st.session_state.get("dialogue_result"))):
        language, level, situation, register = st.columns(4)
        with language:
            dlg_language = st.selectbox("Ngôn ngữ", ["Tiếng Nhật", "Tiếng Anh"], key="dlg_lang")
        with level:
            levels = ["Sơ cấp (N5-N4 / A1-A2)", "Trung cấp (N3 / B1-B2)", "Cao cấp (N2-N1 / C1)"]
            dlg_level = st.selectbox("Cấp độ", levels, key="dlg_level")
        with situation:
            dlg_situation = st.selectbox("Tình huống", ["Tự nhiên / Thông thường", "Nhà hàng / Quán ăn", "Bệnh viện / Nhà thuốc", "Phỏng vấn xin việc", "Mua sắm / Siêu thị", "Gọi taxi / Giao thông", "Khách sạn / Du lịch"], key="dlg_situation")
        with register:
            registers = ["Lịch sự (です/ます)", "Thân mật (タメ口)", "Trang trọng", "Kính ngữ theo vai (尊敬語 / 謙譲語)"] if dlg_language == "Tiếng Nhật" else ["Thân mật (Casual)", "Lịch sự (Polite)", "Trang trọng (Formal)"]
            dlg_register = st.selectbox("Phong cách", registers, key="dlg_politeness")

        topic = st.text_input("Chủ đề", key="dlg_topic_input", placeholder="Ví dụ: đổi món vì dị ứng hải sản")
        role_a, role_b = st.columns(2)
        with role_a:
            person_a = st.text_input("Nhân vật A", value="Khách hàng", key="dlg_role_a")
        with role_b:
            person_b = st.text_input("Nhân vật B", value="Nhân viên", key="dlg_role_b")
        scenario = st.text_area("Diễn biến mong muốn", key="dlg_scenario_input", placeholder="A cần giải quyết một vấn đề thực tế; B hỏi lại, đưa lựa chọn và chốt giải pháp.")
        vocab_input = st.text_area("Từ vựng muốn luyện, mỗi dòng một mục", key="dlg_vocab_input", height=90)
        grammar_input = st.text_area("Ngữ pháp/cấu trúc muốn luyện, mỗi dòng một mục", key="dlg_grammar_input", height=90)

        buttons = st.columns(3)
        if buttons[0].button("Gợi ý chủ đề", key="btn_suggest_topic"):
            with st.spinner("Đang gợi ý chủ đề..."):
                st.session_state.suggested_topics = suggest_topics(dlg_language, dlg_level, st.session_state.get("recent_topics", []))
        if buttons[1].button("Gợi ý mục tiêu học", key="btn_suggest_targets", disabled=not topic):
            with st.spinner("Đang gợi ý mục tiêu..."):
                st.session_state.suggested_vocab_grammar = suggest_vocab_grammar(topic, dlg_language, dlg_level)
        generate_clicked = buttons[2].button("Tạo hội thoại", key="btn_generate_dialogue", type="primary", disabled=not topic)

        if st.session_state.get("suggested_topics"):
            for row in st.session_state.suggested_topics:
                if st.button(row["topic"], key=f"topic_{row['topic']}"):
                    st.session_state["dlg_topic_input"] = row["topic"]
                    st.rerun()
        suggestions = st.session_state.get("suggested_vocab_grammar") or {}
        if suggestions:
            st.caption("Gợi ý chỉ được áp dụng khi bạn bấm nút bên dưới.")
            st.write("Từ vựng: " + "; ".join(suggestions.get("vocab", [])))
            st.write("Cấu trúc: " + "; ".join(suggestions.get("grammar", [])))
            if st.button("Đưa gợi ý vào mục tiêu học", key="apply_suggestions"):
                st.session_state["dlg_vocab_input"] = "\n".join(item.split(" : ", 1)[0] for item in suggestions.get("vocab", []))
                st.session_state["dlg_grammar_input"] = "\n".join(item.split(" : ", 1)[0] for item in suggestions.get("grammar", []))
                st.rerun()

        if generate_clicked:
            scenario_with_roles = f"Nhân vật A: {person_a}. Nhân vật B: {person_b}. {scenario}".strip()
            vocab = [row.strip() for row in vocab_input.splitlines() if row.strip()]
            grammar = [row.strip() for row in grammar_input.splitlines() if row.strip()]
            with st.spinner("Đang tạo hội thoại và kiểm tra chất lượng..."):
                try:
                    result = generate_dialogue(topic, dlg_language, vocab, grammar, dlg_level, dlg_situation, dlg_register, scenario_with_roles, model_name=model_name, reasoning_effort=reasoning_effort)
                    st.session_state.dialogue_result = result
                    st.session_state.dialogue_quiz = generate_pure_quiz(result)
                    st.session_state.dialogue_quiz_results = {}
                    st.session_state.dialogue_history_id = save_dialogue_session(result, session_id=session_id)
                    st.session_state.recent_topics = (st.session_state.get("recent_topics", []) + [topic])[-10:]
                except Exception as exc:
                    st.error(f"Không thể tạo hội thoại: {exc}")

    result = st.session_state.get("dialogue_result")
    if result:
        is_english = _is_english(result.get("language"))
        tts_lang = "en" if is_english else "ja"
        controls = st.columns(4)
        with controls[0]:
            show_hiragana = st.toggle("Hiển thị cách đọc", value=not is_english, key="dlg_show_hira", disabled=is_english)
        with controls[1]:
            show_translation = st.toggle("Hiển thị dịch Việt", value=True, key="dlg_show_vi")
        with controls[2]:
            enable_tts = st.toggle("Bật âm thanh", value=False, key="dlg_enable_tts")
        with controls[3]:
            tts_slow = st.toggle("Nói chậm", value=False, key="dlg_tts_slow", disabled=not enable_tts)

        quality = result.get("quality") or {}
        st.caption(f"Model: {result.get('model_used') or 'mặc định'} · Chất lượng: {quality.get('quality_score', 'N/A')}/100")
        if quality.get("issues"):
            st.warning("Kết quả còn điểm cần xem lại: " + " ".join(quality["issues"]))
        if result.get("quality_repair_error"):
            st.caption("Không thể sửa tự động: " + result["quality_repair_error"])
        usage = result.get("usage") or {}
        if usage:
            cost = estimate_cost(usage, result.get("model_used") or model_name or "gemini-3.5-flash", billing_tier)
            tokens = int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)
            cost_jpy = float(cost.get("total_cost_usd", 0) or 0) * float(usd_to_jpy or 155)
            st.caption(f"Token: {tokens} · Chi phí ước tính: {cost_jpy:.2f} JPY")

        roles = _role_map(result)
        role_text = " · ".join(f"{key}: {value.get('name') or key} ({value.get('role') or value.get('relationship') or 'theo ngữ cảnh'})" for key, value in roles.items())
        if role_text:
            st.info(role_text)
        if result.get("scenario"):
            scenario = result["scenario"]
            st.caption("Mạch hội thoại: " + " → ".join(value for value in [scenario.get("opening"), scenario.get("goal"), scenario.get("problem"), scenario.get("resolution")] if value))

        if enable_tts:
            full_text = "\n".join(f"{turn.get('speaker')}:{turn.get('text')}" for turn in result.get("dialogue", []))
            full_key = "tts_full_" + get_audio_cache_key(full_text, tts_lang, tts_slow)
            if st.button("Nghe toàn bộ hội thoại", key="btn_play_all_tts"):
                with st.spinner("Đang tạo audio hợp lệ..."):
                    audio = generate_full_dialogue_audio(result.get("dialogue", []), lang=tts_lang, slow=tts_slow)
                    if audio:
                        st.session_state[full_key] = audio
                    else:
                        st.warning(get_last_tts_error() or "Không thể tạo audio toàn bài. Bạn vẫn có thể nghe từng câu.")
            if st.session_state.get(full_key):
                st.audio(st.session_state[full_key], format="audio/mp3")

        read_tab, shadow_tab, roleplay_tab, targets_tab, quiz_tab = st.tabs(["Đọc và nghe", "Nhại theo câu", "Đóng vai", "Mục tiêu học", "Kiểm tra"])
        with read_tab:
            for index, turn in enumerate(result.get("dialogue", [])):
                _render_turn(turn, index, roles, show_hiragana=show_hiragana, show_translation=show_translation, enable_tts=enable_tts, tts_lang=tts_lang, tts_slow=tts_slow)
        with shadow_tab:
            st.caption("Nghe từng câu, nhại lại, rồi mở phần dịch và cách nói khác để đối chiếu.")
            for index, turn in enumerate(result.get("dialogue", [])):
                st.markdown(f"**{index + 1}. {roles.get(turn.get('speaker'), {}).get('name') or turn.get('speaker')}**")
                st.write(turn.get("text"))
                if show_hiragana and turn.get("text_hira"):
                    st.caption(turn["text_hira"])
                if enable_tts:
                    key = "shadow_" + get_audio_cache_key(turn.get("text", ""), tts_lang, tts_slow, turn.get("speaker", "A"))
                    if st.button("Nghe để nhại", key=f"shadow_button_{index}"):
                        audio = text_to_speech(turn.get("text", ""), lang=tts_lang, slow=tts_slow, speaker=turn.get("speaker", "A"))
                        if audio:
                            st.session_state[key] = audio
                    if st.session_state.get(key):
                        st.audio(st.session_state[key], format="audio/mp3")
                with st.expander("Đối chiếu sau khi nhại"):
                    st.write("Dịch: " + (turn.get("text_vi") or ""))
                    if turn.get("alternative_expression"):
                        st.write("Cách nói khác: " + turn["alternative_expression"])
        with roleplay_tab:
            learner_role = st.radio("Bạn muốn đóng vai", ["A", "B"], horizontal=True, key="dialogue_learner_role")
            st.caption("Đọc lời của đối phương, tự nói hoặc nhập câu trả lời, rồi mở câu mẫu để so sánh.")
            for index, turn in enumerate(result.get("dialogue", [])):
                if turn.get("speaker") == learner_role:
                    st.write(f"Đến lượt bạn ({roles.get(learner_role, {}).get('name') or learner_role}).")
                    st.text_input("Câu bạn nói", key=f"roleplay_answer_{turn.get('id', index)}")
                    with st.expander("Câu mẫu", expanded=False):
                        st.write(turn.get("text", ""))
                        st.caption(turn.get("text_vi", ""))
                else:
                    _render_turn(turn, index, roles, show_hiragana=show_hiragana, show_translation=show_translation, enable_tts=enable_tts, tts_lang=tts_lang, tts_slow=tts_slow)
        with targets_tab:
            for category, label in (("vocabulary", "Từ vựng"), ("grammar", "Ngữ pháp / cấu trúc")):
                rows = [row for row in result.get("learning_targets", []) if row.get("type") == category]
                if rows:
                    st.markdown(f"#### {label}")
                    for row in rows:
                        status = "Đã xác minh" if row.get("covered") else "Cần kiểm tra lại"
                        st.write(f"{row.get('term')} · {status}")
                        if row.get("realized_form"):
                            st.caption("Dạng dùng: " + row["realized_form"])
                        if row.get("explanation_vi"):
                            st.caption(row["explanation_vi"])
            if result.get("summary"):
                with st.expander("Tóm tắt bài học"):
                    st.write(result["summary"])
            if result.get("notes"):
                with st.expander("Ghi chú ngữ cảnh"):
                    st.write(result["notes"])
        with quiz_tab:
            _render_quiz(st.session_state.get("dialogue_quiz") or generate_pure_quiz(result), is_english, session_id)

        st.divider()
        actions = st.columns(4)
        if actions[0].button("Tạo biến thể khác", key="btn_generate_variation"):
            with st.spinner("Đang tạo biến thể và kiểm tra chất lượng..."):
                try:
                    variation = generate_variation(result, model_name=model_name, reasoning_effort=reasoning_effort)
                    st.session_state.dialogue_result = variation
                    st.session_state.dialogue_quiz = generate_pure_quiz(variation)
                    st.session_state.dialogue_quiz_results = {}
                    st.session_state.dialogue_history_id = save_dialogue_session(variation, session_id=session_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể tạo biến thể: {exc}")
        stem = _safe_filename(result.get("topic", "hoi-thoai"))
        actions[1].download_button("Tải Word", export_dialogue_to_docx(result), f"dialogue_{stem}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        actions[2].download_button("Tải Text", export_dialogue_to_text(result), f"dialogue_{stem}.txt", "text/plain")
        actions[3].download_button("Tải JSON", export_dialogue_to_json(result), f"dialogue_{stem}.json", "application/json")

    _render_history(session_id)
