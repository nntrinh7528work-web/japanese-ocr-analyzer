import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from modules import ocr_engine, text_analyzer
from modules.doc_exporter import export_to_docx
from modules.image_processor import process_image
from modules.multi_image_workflow import add_image_items


OCR_RESPONSE = """TEXT_DIRECTION: horizontal
TEXT_REGIONS: 1
HAS_FURIGANA: no
CONFIDENCE: high
---OCR_START---
The team decided to carry out the plan.
---OCR_END---
---NOTES_START---
none
---NOTES_END---
"""

ANALYSIS_RESPONSE = """## 1. Text Confirmation &amp; Summary
**Original text:**
The team decided to carry out the plan.

**OCR Corrections:** None

**Summary:** The sentence describes a team deciding to execute a plan. The tone is neutral and practical.

## 2. Vocabulary Analysis
### 2.1 All Vocabulary
| # | Word | Base Form | Part of Speech | Vietnamese Meaning | CEFR Level |
|---|------|-----------|----------------|--------------------|------------|
| 1 | team | team | noun | đội nhóm | A2 |
| 2 | decided | decide | verb | quyết định | B1 |
### 2.2 Key Vocabulary (B2-C2, idioms, academic, domain-specific)
| Word | Base Form | Part of Speech | Vietnamese Meaning | Example from Text | Difficulty |
|------|-----------|----------------|--------------------|-------------------|------------|
| carry out | carry out | phrasal verb | thực hiện | "carry out the plan" | B2 |
## 3. Phrasal Verbs &amp; Collocations
| Phrase | Type | Vietnamese Meaning | Example from Text | Note |
|--------|------|--------------------|-------------------|------|
| carry out | phrasal verb | thực hiện | "carry out the plan" | transitive |
## 4. Linking Words &amp; Discourse Markers
| Word/Phrase | Function | Vietnamese Meaning | Example from Text | Register | Difficulty |
|-------------|----------|--------------------|-------------------|----------|------------|
| -- | -- | -- | Không có | -- | -- |
## 5. Grammar Points
**[Past simple]**
- Rule: Use past simple for completed past actions.
- Example from text: "decided"
- Explanation: It presents the decision as completed.
## 6. Sentence Patterns &amp; Structures
**Pattern:** `subject + verb + infinitive`
- Example from text: "decided to carry out"
- Explanation: The infinitive phrase completes the verb "decided".
## 7. Full Analysis Report (Markdown)
# English Text Analysis Report
## 1. Text Overview
The team decided to carry out the plan.
## 2. Vocabulary
See sections above.
## 3. Phrasal Verbs &amp; Collocations
See section 3.
## 4. Linking Words &amp; Discourse Markers
See section 4.
## 5. Grammar Points
See section 5.
## 6. Sentence Patterns
See section 6.
## 7. Overall Assessment
- **CEFR Level Estimate:** B1
- **Text Type:** conversational
- **Dominant Tense:** simple past
- **Writing Style Notes:** neutral
- **Study Recommendations:** Review phrasal verbs.
"""


def sample_image():
    image = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(image).text((40, 120), "English OCR", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def test_full_pipeline(monkeypatch):
    class OcrModel:
        def generate_content(self, _content):
            return SimpleNamespace(text=OCR_RESPONSE, usage_metadata=None)

    class AnalysisModel:
        def generate_content(self, _prompt, generation_config):
            return SimpleNamespace(text=ANALYSIS_RESPONSE, usage_metadata=None)

    monkeypatch.setattr(ocr_engine, "init_gemini", lambda: OcrModel())
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: AnalysisModel())

    image_result = process_image(sample_image())
    ocr_result = ocr_engine.run_ocr(image_result["processed_image_bytes"], image_result["report"])
    analysis = text_analyzer.run_analysis(ocr_result["clean_text"], ocr_result["ocr_notes"])
    docx_bytes = export_to_docx(analysis["full_markdown"])

    assert len(ocr_result["clean_text"]) > 5
    assert analysis["summary"]
    assert analysis["vocabulary_all"]
    assert analysis["phrasal_collocations"]
    assert len(docx_bytes) > 1000


def test_two_images_ocr_and_combined_analysis(monkeypatch):
    class OcrModel:
        def generate_content(self, _content):
            return SimpleNamespace(text=OCR_RESPONSE, usage_metadata=None)

    class AnalysisModel:
        prompts = []

        def generate_content(self, prompt, generation_config):
            self.prompts.append(prompt)
            return SimpleNamespace(text=ANALYSIS_RESPONSE, usage_metadata=None)

    analysis_model = AnalysisModel()
    monkeypatch.setattr(ocr_engine, "init_gemini", lambda: OcrModel())
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: analysis_model)

    items, added, errors = add_image_items(
        [],
        [("page-1.jpg", sample_image()), ("page-2.jpg", sample_image() + b"different")],
    )
    assert added == ["page-1.jpg", "page-2.jpg"]
    assert errors == []

    for item in items:
        item["ocr_result"] = ocr_engine.run_ocr(item["processed_image_bytes"], item["report"])
        item["edited_text"] = item["ocr_result"]["clean_text"]

    pages = [
        {
            "page_index": index,
            "page_name": item["name"],
            "text": item["edited_text"],
            "notes": item["ocr_result"].get("ocr_notes", []),
        }
        for index, item in enumerate(items, 1)
    ]
    analysis = text_analyzer.run_page_analyses(pages)
    assert analysis["summary"]
    assert len(analysis["page_analyses"]) == 2
    assert len(analysis_model.prompts) == 2
    # With concurrent execution prompts may arrive in any order.
    all_prompts = "\n".join(analysis_model.prompts)
    assert "page-1.jpg" not in all_prompts  # page names are not sent as part of the prompt
    assert "=== ẢNH 2: page-2.jpg ===" not in all_prompts
