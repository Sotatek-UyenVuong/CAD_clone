# CAD Pipeline - Current Technical Context

File này tóm tắt trạng thái runtime hiện tại sau khi chuyển sang kiến trúc `single orchestrator + pure executors`.

## 1) Mục tiêu hệ thống

- Index tài liệu CAD (PDF/ảnh; DXF khi cần cho count).
- Trả lời Q&A có grounding theo trang/tài liệu.
- Search hiện tại: title-block deterministic + lexical Mongo.
- Không còn vector embedding / Qdrant.

## 2) Module chính (đang dùng)

```text
cad_pipeline/
  api/app.py
  pipeline/
    upload_pipeline.py
    qa_orchestrator_pipeline.py
    search_pipeline.py
    delete_pipeline.py
  prompts/
    qa_orchestrator_prompts.py
    agent_prompts.py
  agents/
    tool_router.py
    file_agent.py
    page_agent.py
    folder_agent.py
  tools/
    search_tool.py
    count_tool.py
    area_tool.py
    report_tool.py
    viz_tool.py
    to_excel.py
  storage/
    mongo.py
    s3_store.py
  config.py
```

Ghi chú:
- API import `run_qa` từ `pipeline/qa_orchestrator_pipeline.py`.
- `qa_pipeline.py` cũ đã bỏ khỏi flow chính.

## 3) Runtime config quan trọng

Từ `config.py`:
- Mongo: `DATABASE_URL`, `DATABASE_NAME`
- Gemini: `GEMINI_API_KEY`
- Marker OCR: `MARKER_API_KEY`
- Search tuning: `TOP_K`, `TOP_N`
- Storage mode:
  - `USE_S3=true`: ảnh/original lên R2
  - `USE_S3=false`: lưu local + serve qua static route
- Agent scope: `AGENT_MAX_PAGES`

## 4) Upload pipeline (`pipeline/upload_pipeline.py`)

Flow:
1. Persist original (S3 hoặc local).
2. Rẽ nhánh theo định dạng:
   - `DWG`: convert `DWG -> DXF` bằng script `tools/_archive/dwg_to_dxf_converter.py`, persist DXF vào `cad_pipeline/data/originals/<file_id>/cad/`, rồi index thành 1 page có `dxf_path`.
   - `DXF`: đi thẳng CAD geometry flow, persist vào thư mục bền vững và index 1 page có `dxf_path`.
   - `DOC/DOCX`: convert sang PDF (LibreOffice headless), sau đó đi theo flow PDF.
   - `XLS/XLSX`: dùng marker text pipeline để tách nội dung theo sheet/page, lưu `context_md` dạng text.
   - `PDF` hoặc ảnh (`png/jpg/jpeg/webp/gif/bmp/tif/tiff`): render thành page images để xử lý layout/page context.
3. Tạo/ensure folder + file records trong MongoDB.
4. Với flow page-based (PDF/ảnh):
   - tạo `image_url`,
   - layout detection (có fallback nếu unavailable),
   - page summary + block processing concurrent,
   - build `context_md`,
   - save page.
5. Kết thúc file:
   - build `files.summary`,
   - build `files.short_summary`,
   - build `files.title_block_index`,
   - rebuild `folders.summary`.

## 5) Q&A orchestrator (`pipeline/qa_orchestrator_pipeline.py`)

### 5.1 Thiết kế
- **Một orchestrator duy nhất (LLM)** để plan/replan.
- **Executors thuần** chỉ thực thi tác vụ:
  - `file_agent`, `page_reason`,
  - `search`, `count`, `area`, `viz`,
  - `report_pdf`, `report_docx`, `report_excel`.
- Không dùng routing tầng agent kiểu cũ để quyết định flow lớn.

### 5.2 Vòng lặp xử lý
1. Load context: history, recent citations, language context, explicit pages, working files, context summary.
2. Orchestrator chọn **1 next action**.
3. Executor chạy action tương ứng.
4. Review-lite tự đánh giá:
   - đủ evidence/citation chưa,
   - có cover explicit pages không,
   - có nên finalize hay chạy step tiếp.
5. Nếu chưa đạt thì replan và lặp (loop guard + max steps), đạt thì finalize.

### 5.3 Language handling
- LLM quyết định `language_context` theo BCP-47 (không giới hạn 3 ngôn ngữ).
- Fallback system messages:
  - ưu tiên vi/ja/en nếu match nhanh,
  - ngôn ngữ khác thì dịch runtime qua `_translate_text`.

### 5.4 Scope guard quan trọng
- Khi user chỉ định explicit pages, tool/page reasoning chạy trong scope nghiêm ngặt.
- `_pick_tool_scope_pages` đã bỏ fallback ngầm kiểu `all_pages[:N]` để tránh trả lời sai do đoán phạm vi.

## 6) Search behavior

### 6.1 `tools/search_tool.py`
- Nếu có ảnh: ưu tiên title-block lookup (`files.title_block_index`).
- Không match thì fallback lexical Mongo (`retrieval_mode=lexical_search`).

### 6.2 `storage/mongo.py` lexical guard
- Đã thêm giới hạn độ dài query/token để tránh query quá dài gây lỗi DB.
- Có xử lý regex compile error để không làm vỡ pipeline.

## 7) Layout detect model (lưu ý vận hành)

- Hệ thống upload PDF chuẩn vẫn dùng `LayoutDetector` để detect block layout theo trang trước khi build `context_md`.
- Dữ liệu/training assets cho layout detect nằm ở phần `layout_detect/` (đã dọn các artifact archive không còn dùng trong runtime chính).
- Khi deploy model mới, cần bảo đảm output class mapping vẫn tương thích với downstream block/page processor để tránh lệch schema context.

## 8) Tool behavior tóm tắt

- `count_tool`: ưu tiên DXF -> image -> context.
- `area_tool`: ưu tiên catalog/unit -> vision -> context.
- `report_tool`: xuất PDF/DOCX/Excel từ kết quả đã có scope.
- Q&A orchestrator có fallback cho action `count`: nếu thiếu page scope nhưng file có `dxf_path`, chạy đếm trực tiếp theo DXF file-level.
- Quan trọng: flow DXF cho `count/area` hiện được tối ưu theo convention bản vẽ kiến trúc Nhật (Japanese CAD drawing conventions + symbol DB tiếng Nhật). Với tài liệu ngoài miền này, độ chính xác có thể giảm và nên fallback thêm vision/context.

## 9) Storage + scope

Collections chính:
- `folders`, `files`, `pages`
- `chat_history`, `chat_sessions`
- `notifications`

Scope behavior:
- `POST /tools/search`: global default, có `folder_id`/`file_id` thì hẹp scope.
- `POST /qa`:
  - có thể chạy nhánh search hoặc orchestrator QA,
  - vẫn tôn trọng session files (`session_file_ids`) nếu có.
- `POST /qa/image`:
  - image-only có thể đi search tool,
  - có query thì vào orchestrator QA.

## 10) Endpoint thường dùng

- Upload: `/upload`, `/upload/{job_id}/status`
- QA: `/qa`, `/qa/stream`, `/qa/jobs`, `/qa/jobs/{job_id}`, `/qa/image`, `/qa/image/stream`
- Search: `/search`, `/tools/search`, `/tools/search/suggest`
- Data: `/folders`, `/folders/{folder_id}/files`, `/files/{file_id}`, `/files/{file_id}/pages`
- Delete: `/files/{file_id}`, `/folders/{folder_id}`
- Chat sessions: `/chat-sessions`, `/chat-sessions/{session_id}`
- Tools: `/tools/count`, `/tools/count/groups`, `/tools/area/units`, `/tools/area/units/{unit_label}`

Lưu ý endpoint `GET /files/{file_id}/original`:
- Với file CAD (`.dwg`/`.dxf`), endpoint ưu tiên trả file từ `dxf_path` (tức bản DXF).
- Nếu không phải CAD hoặc không có `dxf_path`, endpoint trả `file_url` như thông thường.

## 11) Đã loại bỏ

- Cohere embedding logic.
- Qdrant storage/search/delete logic.
- `cad_pipeline/agents/language_utils.py`.
- Flow `qa_pipeline.py` cũ dựa trên multi-agent routing tầng cao.

---

Khi đổi flow/prompt của orchestrator, cập nhật file này trước để giữ đồng bộ nhận thức giữa code và tài liệu.
