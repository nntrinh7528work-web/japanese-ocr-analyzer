"""Dialogue Practice page view component."""

from __future__ import annotations
import streamlit as st

from modules.dialogue_exporter import (
    export_dialogue_to_docx,
    export_dialogue_to_json,
    export_dialogue_to_text,
)
from modules.dialogue_generator import generate_dialogue, generate_variation, suggest_topics
from modules.dialogue_history import (
    get_due_sm2_cards,
    get_practice_history,
    get_streak_days,
    save_dialogue_session,
    update_sm2_card,
)
from modules.dialogue_quiz import generate_pure_quiz
from modules.tts_engine import text_to_speech


def render_dialogue_tab() -> None:
    """Render Tab 2: Dialogue Practice & Spaced Repetition (SM-2)."""
    st.subheader("💬 Luyện Hội Thoại Hằng Ngày")

    # ── Display streak & due SM-2 cards notice ──
    streak = get_streak_days()
    due_cards = get_due_sm2_cards()
    st.markdown(
        f"""<div class="streak-banner">
            <span class="streak-number">🔥 {streak}</span>
            <span>ngày liên tục</span>
            <span style="margin-left: auto;">🧠 <strong>{len(due_cards)}</strong> từ cần ôn hôm nay</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Spaced Repetition Section ──
    if due_cards:
        with st.expander("🧠 Ôn tập Spaced Repetition (SM-2)", expanded=False):
            st.write("Đánh giá độ nhớ của bạn cho các từ/cấu trúc cần ôn hôm nay:")
            for card in due_cards[:5]:
                st.markdown(f"### 🎴 `{card['word']}`")
                st.caption(f"Gợi ý / Nghĩa: {card['meaning']}")
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("❌ 0: Quên", key=f"sm2_0_{card['id']}"):
                    update_sm2_card(card["id"], 0)
                    st.toast("Đã ghi nhận: Cần luyện lại sớm")
                    st.rerun()
                if c2.button("⚠️ 3: Khó", key=f"sm2_3_{card['id']}"):
                    update_sm2_card(card["id"], 3)
                    st.toast("Đã ghi nhận: Khó nhớ")
                    st.rerun()
                if c3.button("👍 4: Tốt", key=f"sm2_4_{card['id']}"):
                    update_sm2_card(card["id"], 4)
                    st.toast("Đã ghi nhận: Tốt")
                    st.rerun()
                if c4.button("🌟 5: Rất dễ", key=f"sm2_5_{card['id']}"):
                    update_sm2_card(card["id"], 5)
                    st.toast("Đã ghi nhận: Rất dễ!")
                    st.rerun()

    # ── Configuration Form ──
    c_lang, c_level, c_sit, c_pol = st.columns(4)
    with c_lang:
        dlg_language = st.selectbox("Ngôn ngữ:", ["Tiếng Nhật", "Tiếng Anh"], key="dlg_lang")
    with c_level:
        dlg_level = st.selectbox("Cấp độ:", ["Sơ cấp", "Trung cấp", "Cao cấp"], key="dlg_level")
    with c_sit:
        dlg_situation = st.selectbox(
            "Tình huống:",
            ["Tự nhiên / Thông thường", "Nhà hàng / Quán ăn", "Bệnh viện / Nhà thuốc", "Phỏng vấn xin việc", "Mua sắm / Siêu thị", "Gọi taxi / Giao thông", "Khách sạn / Du lịch"],
            key="dlg_situation",
        )
    with c_pol:
        dlg_politeness = st.selectbox(
            "Phong cách / Kính ngữ:",
            ["Lịch sự (です/ます)", "Thân mật (タメ口)", "Kính ngữ (敬語)", "Khiêm nhường ngữ (謙醸語)"],
            key="dlg_politeness",
        )

    # ── Display Toggles ──
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        show_hiragana = st.toggle("Hiển thị Hiragana", value=True, key="dlg_show_hira")
    with col_t2:
        show_translation = st.toggle("Hiển thị dịch tiếng Việt", value=True, key="dlg_show_vi")
    with col_t3:
        enable_tts = st.toggle("🔊 Phát âm thanh", value=False, key="dlg_enable_tts")
    with col_t4:
        tts_slow = st.toggle("🐢 Nói chậm", value=False, key="dlg_tts_slow", disabled=not enable_tts)

    if st.button("🎲 Gợi ý chủ đề hôm nay", key="btn_suggest_topic"):
        with st.spinner("Đang tìm chủ đề..."):
            topics = suggest_topics(dlg_language, dlg_level, st.session_state.recent_topics)
            st.session_state.suggested_topics = topics

    if st.session_state.get("suggested_topics"):
        for t in st.session_state.suggested_topics:
            if st.button(f"📌 {t['topic']}", key=f"topic_{t['topic']}"):
                st.session_state.selected_topic = t["topic"]

    topic_input = st.text_input(
        "Hoặc nhập chủ đề của riêng bạn:",
        value=st.session_state.get("selected_topic", ""),
        key="dlg_topic_input",
    )

    scenario_input = st.text_area(
        "Miêu tả diễn biến / trường hợp cụ thể (định hướng hội thoại):",
        placeholder="Ví dụ: A là khách muốn đổi món do bị dị ứng hải sản, B là bồi bàn lịch sự xin lỗi và gợi ý món thay thế.",
        help="Nhập hoàn cảnh hoặc hướng diễn biến cụ thể bạn muốn cuộc hội thoại diễn ra",
        key="dlg_scenario_input",
        height=80,
    )

    vocab_input = st.text_area(
        "Từ vựng muốn luyện (mỗi từ 1 dòng, có thể để trống):",
        key="dlg_vocab_input",
        height=80,
    )
    grammar_input = st.text_area(
        "Cấu trúc ngữ pháp muốn luyện (mỗi cấu trúc 1 dòng, có thể để trống):",
        key="dlg_grammar_input",
        height=80,
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        gen_clicked = st.button("✨ Tạo hội thoại", key="btn_generate_dialogue", disabled=not topic_input, type="primary")
    with btn_col2:
        var_clicked = st.button("🔄 Tạo biến thể khác", key="btn_generate_variation", disabled=not st.session_state.get("dialogue_result"))

    if gen_clicked:
        vocab_list = [v.strip() for v in vocab_input.splitlines() if v.strip()]
        grammar_list = [g.strip() for g in grammar_input.splitlines() if g.strip()]
        with st.spinner("Đang tạo hội thoại..."):
            try:
                result = generate_dialogue(
                    topic_input, dlg_language, vocab_list, grammar_list, dlg_level, dlg_situation, dlg_politeness, scenario_input
                )
                st.session_state.dialogue_result = result
                st.session_state.dialogue_quiz = generate_pure_quiz(result)
                st.session_state.recent_topics.append(topic_input)
                save_dialogue_session(result)
                if not result["fully_covered"]:
                    st.warning("Một số từ/ngữ pháp chưa được dùng hết, nhưng đây là bản tốt nhất.")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    if var_clicked and st.session_state.get("dialogue_result"):
        with st.spinner("Đang tạo biến thể mới..."):
            try:
                var_result = generate_variation(st.session_state.dialogue_result)
                st.session_state.dialogue_result = var_result
                st.session_state.dialogue_quiz = generate_pure_quiz(var_result)
                save_dialogue_session(var_result)
                st.success("Đã tạo biến thể mới!")
            except Exception as e:
                st.error(f"Lỗi tạo biến thể: {e}")

    # ── Dialogue Output Display ──
    if st.session_state.dialogue_result:
        r = st.session_state.dialogue_result
        st.markdown(f"### 📖 Chủ đề: {r['topic']}")
        meta_info = f"Tình huống: {r.get('situation', 'Thông thường')} | Kính ngữ: {r.get('politeness_level', 'Lịch sự')}"
        if r.get("scenario_description"):
            meta_info += f" | Miêu tả hoàn cảnh: _{r['scenario_description']}_"
        st.caption(meta_info)

        # TTS: Play all button
        if enable_tts:
            if st.button("🔊 Phát toàn bộ hội thoại", key="btn_play_all_tts"):
                with st.spinner("Đang tạo audio..."):
                    all_text = "\n".join(t["text"] for t in r["dialogue"] if t.get("text"))
                    full_audio = text_to_speech(all_text, lang="ja", slow=tts_slow)
                    if full_audio:
                        st.session_state["tts_full_audio"] = full_audio
            if st.session_state.get("tts_full_audio"):
                st.audio(st.session_state["tts_full_audio"], format="audio/mp3")

        # Dialogue Turns
        for turn_idx, turn in enumerate(r["dialogue"]):
            speaker_class = "speaker-bubble-a" if turn["speaker"] == "A" else "speaker-bubble-b"
            icon = "🗣️" if turn["speaker"] == "A" else "💭"
            
            st.markdown(
                f'<div class="{speaker_class}"><strong>{icon} {turn["speaker"]}:</strong> {turn["text"]}</div>',
                unsafe_allow_html=True,
            )

            if show_hiragana and turn.get("text_hira"):
                st.caption(f"_{turn['text_hira']}_")

            # TTS per turn button
            if enable_tts:
                tts_cache_key = f"tts_audio_{turn_idx}_{tts_slow}"
                if st.button(f"🔊 Nghe câu {turn['speaker']} ({turn_idx + 1})", key=f"tts_{turn_idx}"):
                    audio_data = text_to_speech(turn["text"], lang="ja", slow=tts_slow)
                    if audio_data:
                        st.session_state[tts_cache_key] = audio_data
                    else:
                        st.warning("Không thể tạo audio. Kiểm tra kết nối mạng.")
                if st.session_state.get(tts_cache_key):
                    st.audio(st.session_state[tts_cache_key], format="audio/mp3")

            is_revealed = st.session_state.revealed_turns.get(turn_idx, False)
            if show_translation or is_revealed:
                st.caption(f"🇻🇳 {turn['text_vi']}")
            else:
                if st.button(f"👁️ Hiện bản dịch câu {turn['speaker']}", key=f"rev_{turn_idx}"):
                    st.session_state.revealed_turns[turn_idx] = True
                    st.rerun()

            if turn["highlights"]:
                st.caption(f"🎯 Dùng: {', '.join(turn['highlights'])}")

        st.divider()

        if r.get("summary"):
            st.markdown("#### 📚 Tóm tắt Từ vựng & Ngữ pháp")
            st.info(r["summary"])

        with st.expander("✅ Kiểm tra độ phủ từ vựng/ngữ pháp"):
            for target, covered in r["coverage_check"].items():
                icon = "✅" if covered else "❌"
                st.markdown(f"{icon} {target}")

        if r.get("notes"):
            st.info(r["notes"])

        # ── MINI QUIZ ──
        st.divider()
        st.markdown("### 📝 Mini Quiz (Luyện tập sau bài học)")
        if st.session_state.dialogue_quiz:
            quiz = st.session_state.dialogue_quiz
            q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs(["1. Điền từ (Cloze)", "2. Trắc nghiệm nghĩa", "3. Sắp xếp câu", "4. Dịch Việt - Nhật"])

            with q_tab1:
                if quiz.get("cloze"):
                    for q in quiz["cloze"]:
                        st.markdown(f"**Câu hỏi:** {q['sentence']}")
                        st.caption(f"Dịch: {q['text_vi']}")
                        user_ans = st.radio(f"Chọn từ thích hợp:", q["options"], key=f"ui_{q['id']}")
                        if st.button("Kiểm tra", key=f"btn_{q['id']}"):
                            if user_ans == q["target_word"]:
                                st.success("🎉 Chính xác!")
                            else:
                                st.error(f"❌ Chưa đúng. Đáp án đúng là: `{q['target_word']}`")
                else:
                    st.info("Không có từ vựng phù hợp cho bài cloze test.")

            with q_tab2:
                if quiz.get("mcq"):
                    for q in quiz["mcq"]:
                        st.markdown(f"**{q['question']}**")
                        user_ans = st.radio("Chọn đáp án:", q["options"], key=f"ui_{q['id']}")
                        if st.button("Kiểm tra", key=f"btn_{q['id']}"):
                            if user_ans == q["correct_answer"]:
                                st.success("🎉 Chính xác!")
                            else:
                                st.error(f"❌ Chưa đúng. Đáp án: `{q['correct_answer']}`")
                else:
                    st.info("Chưa có trắc nghiệm từ vựng.")

            with q_tab3:
                if quiz.get("reorder"):
                    for q in quiz["reorder"]:
                        st.markdown(f"**Ý nghĩa câu:** {q['text_vi']}")
                        st.write("Từ gợi ý: " + "  |  ".join(f"`{t}`" for t in q["shuffled_tokens"]))
                        user_input = st.text_input("Nhập lại câu hoàn chỉnh:", key=f"ui_{q['id']}")
                        if st.button("Kiểm tra", key=f"btn_{q['id']}"):
                            if user_input.strip() == q["original"].strip():
                                st.success("🎉 Bạn xếp đúng rồi!")
                            else:
                                st.info(f"Đáp án gốc: `{q['original']}`")
                else:
                    st.info("Không có câu sắp xếp.")

            with q_tab4:
                if quiz.get("translate"):
                    for q in quiz["translate"]:
                        st.markdown(f"**Dịch sang tiếng Nhật:** _{q['prompt_vi']}_")
                        user_trans = st.text_input("Câu trả lời của bạn:", key=f"ui_{q['id']}")
                        if st.button("Xem đáp án", key=f"btn_{q['id']}"):
                            st.info(f"Câu gốc: `{q['correct_jp']}` ({q['text_hira']})")

        # ── EXPORT BUTTONS ──
        st.divider()
        st.markdown("### 💾 Export bài luyện hội thoại")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            docx_data = export_dialogue_to_docx(r)
            st.download_button(
                "📝 Tải file Word (.docx)",
                data=docx_data,
                file_name=f"dialogue_{r['topic']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="btn_dl_docx",
            )
        with exp_col2:
            txt_data = export_dialogue_to_text(r)
            st.download_button(
                "📄 Tải file Text (.txt)",
                data=txt_data,
                file_name=f"dialogue_{r['topic']}.txt",
                mime="text/plain",
                key="btn_dl_txt",
            )
        with exp_col3:
            json_data = export_dialogue_to_json(r)
            st.download_button(
                "📋 Tải JSON (.json)",
                data=json_data,
                file_name=f"dialogue_{r['topic']}.json",
                mime="application/json",
                key="btn_dl_json",
            )

    # ── PRACTICE HISTORY LOG ──
    st.divider()
    st.markdown("### 📜 Lịch sử luyện tập gần đây")
    history_logs = get_practice_history(limit=10)
    if history_logs:
        st.dataframe(
            history_logs,
            column_config={
                "id": "ID",
                "created_at": "Thời gian",
                "topic": "Chủ đề",
                "language": "Ngôn ngữ",
                "level": "Cấp độ",
                "situation": "Tình huống",
                "politeness_level": "Kính ngữ",
                "scenario_description": "Miêu tả hoàn cảnh",
                "quiz_score": "Điểm Quiz",
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Chưa có lịch sử luyện tập nào được lưu.")
