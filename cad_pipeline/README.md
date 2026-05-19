# CAD Pipeline

Pipeline index + Q&A cho bản vẽ CAD kiến trúc (PDF/ảnh, có DXF cho tool đếm).

## 1) Kiến trúc hiện tại

```text
UPLOAD
file -> render pages -> layout detect -> OCR/vision per block-group
     -> build context_md + summaries + title_block_index -> MongoDB

Q&A (single orchestrator)
query + history + files + citations + context_summary
  -> orchestrator chọn 1 action
  -> executor chạy action
  -> review-lite finalize hoặc replan

SEARCH TOOL
/tools/search: title-block lookup (nếu có ảnh) -> lexical Mongo fallback
```

Trạng thái mới:
- Đã bỏ hoàn toàn vector embedding / Qdrant.
- `run_qa` dùng `pipeline/qa_orchestrator_pipeline.py`.
- Orchestrator duy nhất điều hướng toàn bộ flow.
- Executors thuần: `file_agent`, `page_reason`, `search`, `count`, `area`, `viz`, `report_*`.
- Prompt tập trung ở `prompts/qa_orchestrator_prompts.py` và `prompts/agent_prompts.py`.

## 2) Setup nhanh

```bash
cd /path/to/CAD_clone
pip install -r requirements.txt
```

Tạo `cad_pipeline/.env`:

```env
DATABASE_URL=mongodb://localhost:27017
DATABASE_NAME=cad_pipeline

GEMINI_API_KEY=...
MARKER_API_KEY=...

USE_S3=false
API_BASE_URL=http://localhost:8001

TOP_K=100
TOP_N=15
PDF_DPI=300
AGENT_MAX_PAGES=25
```

Ghi chú:
- Marker OCR dùng Datalab Marker API (`MARKER_API_KEY`).
- Khi `USE_S3=false`, ảnh được serve local qua `/images`.

## 3) Chạy backend

```bash
uvicorn cad_pipeline.api.app:app --host 0.0.0.0 --port 8001 --reload
```

## 4) Chạy full stack ổn định bằng tmux

```bash
# Backend
tmux new-session -d -s cad_clone_api \
  'cd /path/to/CAD_clone && python -m uvicorn cad_pipeline.api.app:app --host 0.0.0.0 --port 8001 --reload'

# Frontend
tmux new-session -d -s cad_clone_ui \
  'cd /path/to/CAD_clone && __VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS=.ngrok-free.dev VITE_API_BASE_URL=same-origin npm --prefix "/path/to/CAD_clone/Chatbotsysteminterface" run dev -- --host 127.0.0.1 --port 5173'

# Ngrok
tmux new-session -d -s cad_clone_ngrok \
  'cd /path/to/CAD_clone && ngrok http 5173 --log stdout'
```

Lệnh hữu ích:
- `tmux ls`
- `tmux attach -t cad_clone_api`
- `tmux attach -t cad_clone_ui`
- `tmux attach -t cad_clone_ngrok`
- Detach không tắt process: `Ctrl+b` rồi `d`

## 5) Core flows

### Upload flow
- Persist original (R2 nếu `USE_S3=true`, local nếu `false`).
- Nếu file là `DWG`:
  - convert sang `DXF` bằng script `tools/_archive/dwg_to_dxf_converter.py`,
  - persist DXF vào `cad_pipeline/data/originals/<file_id>/cad/<file_stem>.dxf`,
  - tạo 1 page context có `dxf_path`,
  - lưu `files.dxf_path` + `pages.dxf_path` để tool đếm dùng trực tiếp.
- Nếu file là `DXF`: đi thẳng flow CAD geometry (không render ảnh/PDF).
- PDF -> page PNG (`pdf_to_images.py`), hoặc marker-text pipeline với file office.
- Per page:
  - layout detect (nếu available),
  - page summary + block processing concurrent,
  - build `context_md`,
  - save page vào Mongo.
- Cuối file:
  - build `files.summary`, `files.short_summary`,
  - build `files.title_block_index`,
  - rebuild `folders.summary`.

### Layout detect model
- Runtime upload flow dùng `LayoutDetector` để detect block layout theo trang trước khi page reasoning.
- Khi retrain/thay model trong `layout_detect/`, cần giữ ổn định class mapping vì downstream parser/context builder phụ thuộc trực tiếp vào nhãn block.

### Q&A flow (`POST /qa`)
- Load history + recent citations + file scope + explicit pages + context summary.
- Orchestrator chọn một action trong:
  - `file_agent`, `page_reason`, `search`, `count`, `area`, `viz`,
  - `report_pdf`, `report_docx`, `report_excel`, `direct_answer`, `finalize`.
- Chạy executor tương ứng.
- Review-lite quyết định finalize hoặc replan step kế tiếp.
- Có loop guard và max steps để tránh lặp vô hạn.
- Riêng action `count`: nếu không có page scope phù hợp nhưng file có `dxf_path`, hệ thống fallback đếm trực tiếp từ DXF.

### Search flow
- `run_search`: lexical retrieval trên `short_summary` + `context_md`, optional scope `folder_id`/`file_id`.
- `run_search_tool`: nếu có ảnh thì ưu tiên title-block lookup; không match mới fallback lexical.
- Đã thêm guard query/token trong Mongo lexical để tránh lỗi query dài.

### Miền áp dụng DXF (quan trọng)
- Tool `count` và nhánh `area` dùng DXF hiện tối ưu cho tài liệu kiến trúc Nhật (naming, symbol groups, text conventions tiếng Nhật).
- Với bản vẽ ngoài miền Nhật, nên kỳ vọng fallback nhiều hơn qua vision/context và kiểm chứng kết quả thủ công khi cần độ chính xác cao.

## 6) Scope behavior (quan trọng)

- `GET /tools/search/suggest`
  - Mặc định: global
  - Có `folder_id`: scope theo folder
- `POST /tools/search`
  - Mặc định: global
  - Có `folder_id`/`file_id`: scope hẹp hơn
- `POST /qa`
  - Có thể đi nhánh search hoặc orchestrator QA
  - Có `session_file_ids` thì giới hạn trong file scope của session
- `POST /qa/image`
  - image-only: có thể gọi `run_search_tool` trực tiếp
  - có query: vào orchestrator `run_qa`

Lưu ý:
- `GET /files/{file_id}/original` sẽ ưu tiên trả bản DXF cho file CAD (`.dwg`/`.dxf`).
- Trình duyệt thường không render DXF inline; tab Original có thể hiện tải xuống thay vì preview trực tiếp.

## 7) API endpoints chính

- `GET /health`
- `POST /upload`
- `GET /upload/{job_id}/status`
- `GET /notifications`
- `PATCH /notifications/{notification_id}/read`
- `PATCH /notifications/read-all`
- `POST /qa`
- `POST /qa/stream`
- `POST /qa/jobs`
- `GET /qa/jobs/{job_id}`
- `POST /qa/image`
- `POST /qa/image/stream`
- `POST /search`
- `POST /tools/search`
- `GET /tools/search/suggest`
- `POST /folders`
- `GET /folders`
- `GET /folders/{folder_id}/files`
- `GET /files/{file_id}`
- `GET /files/{file_id}/pages`
- `GET /files/{file_id}/pages/{page_number}`
- `DELETE /files/{file_id}`
- `DELETE /folders/{folder_id}`
- `GET /chat-sessions`
- `GET /chat-sessions/{session_id}`
- `DELETE /chat-sessions/{session_id}`
- `GET /tools/count/groups`
- `GET /tools/count`
- `GET /tools/area/units`
- `GET /tools/area/units/{unit_label}`
- `POST /tools/count/context`
- `POST /tools/area/context`

## 8) Ví dụ nhanh

Upload:

```bash
curl -X POST http://localhost:8001/upload \
  -F "file=@drawing.pdf" \
  -F "folder_id=folder_001" \
  -F "folder_name=Electrical Drawings"
```

Q&A:

```bash
curl -X POST http://localhost:8001/qa \
  -H "Content-Type: application/json" \
  -d '{"query":"EV co bao nhieu cai?","folder_id":"folder_001"}'
```

Search:

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query":"stair core 9F","folder_id":"folder_001"}'
```

## 9) Cấu trúc thư mục chính

```text
cad_pipeline/
  api/app.py
  pipeline/{upload_pipeline,qa_orchestrator_pipeline,search_pipeline,delete_pipeline}.py
  prompts/{qa_orchestrator_prompts,agent_prompts}.py
  agents/{folder_agent,file_agent,page_agent,tool_router}.py
  tools/{search_tool,count_tool,area_tool,report_tool,viz_tool}.py
  core/{pdf_to_images,layout_detect,page_processor,context_builder,marker_pdf}.py
  storage/{mongo,s3_store}.py
  config.py
```
