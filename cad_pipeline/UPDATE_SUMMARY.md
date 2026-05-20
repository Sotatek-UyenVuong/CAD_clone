# CAD Update Summary (Current Runtime)

Tai lieu nay tong hop cac diem moi cua he thong CAD theo dung code hien tai.

## 1) QA orchestrator-first architecture

- Luong QA da chuyen sang `single orchestrator + pure executors` trong `pipeline/qa_orchestrator_pipeline.py`.
- Orchestrator chay vong lap:
  - load context (history, citations, files, explicit pages, language),
  - plan 1 action,
  - execute action,
  - review-lite (evidence/citation/page coverage),
  - finalize hoac replan.
- Actions duoc ho tro:
  - `file_agent`, `page_reason`, `search`, `count`, `area`, `viz`,
  - `report_pdf`, `report_docx`, `report_excel`,
  - `direct_answer`, `finalize`.
- `tool_result` co `orchestrator_trace` de theo doi tung step.

## 2) CAD upload flow (DWG/DXF)

- Backend upload nhan `.dwg` va `.dxf` (`api/app.py` + `upload_pipeline.py`).
- Neu upload la DWG:
  - `upload_pipeline.py` goi `_convert_dwg_to_dxf(...)`,
  - ham nay dynamic-load va goi truc tiep `convert(...)` trong `tools/_archive/dwg_to_dxf_converter.py`.
- Neu upload la DXF:
  - bo qua convert, di thang vao DXF indexing pipeline.
- DXF artifact duoc persist o duong dan ben vung:
  - `cad_pipeline/data/originals/<file_id>/cad/<file_stem>.dxf`.
- CAD file duoc index thanh 1 page context co `dxf_path` de QA/tools co the su dung ngay.

## 3) DXF catalog at ingestion time

- Luc upload CAD, he thong parse DXF bang `ezdxf`:
  - quet `INSERT` de lay symbol blocks + count,
  - quet `TEXT/MTEXT` de lay unit/text labels + count (co loc text nhieu nhieu de dung cho hoi dap).
- Catalog duoc ghi vao:
  - `page short_summary`,
  - `page context_md`,
  - `file summary`.
- Da bo sung map alias cho block tho (`ZGroup...`) bang cach:
  - doc text trong block definition (co nested insert, gioi han depth),
  - chon alias/samples de hien thi de hieu hon.

## 4) CAD preview behavior (Page Summary centric)

- DXF page duoc luu voi `blocks=[]` (khong co block crop tu anh/PDF), nen preview CAD khong dua vao block detector.
- Phan "preview" cho CAD page duoc dua tren:
  - `short_summary`: tong quan nhanh (so luong symbol, so unit/text labels, top labels),
  - `context_md`: catalog chi tiet (top symbol blocks + unit/text labels + sample).
- Nghia la voi file DWG/DXF, nguon du lieu preview/hieu nghia page la "Page Summary + context_md catalog", khong phai image block OCR.

## 5) Count tool updates

- `count_tool` uu tien DXF:
  1. `dxf_exact` (symbol DB + INSERT),
  2. `dxf_symbol_dict`,
  3. `dxf_text_label`.
- Co scope theo viewport/layout khi co thong tin.
- Orchestrator co fallback:
  - neu action `count` khong co page scope hop le nhung file co `dxf_path`,
    van chay count truc tiep tren DXF file-level.

## 6) Area tool updates

- `area_tool` ho tro 3 mode:
  - unit catalog lookup (`unit_room_catalog.json`),
  - context extraction (`context_md`),
  - vision extraction (image).
- Output thong nhat de orchestration de finalize/replan.

## 7) Report export tools

- `run_report_pdf`:
  - markdown -> PDF, fallback markdown neu convert fail.
- `run_report_docx`:
  - markdown -> DOCX (template), fallback markdown neu fail.
- `run_report_excel`:
  - Gemini tao JSON table schema tu query/pages/tool_result/chat_history,
  - xuat workbook gom Summary, data sheet, Source Pages (neu co).

## 8) Original file serving for CAD

- Endpoint `GET /files/{file_id}/original` uu tien tra DXF cho file CAD (`.dwg/.dxf`) neu co `dxf_path`.
- FE da cap nhat:
  - cho phep upload `.dwg/.dxf`,
  - dong bo ten file download theo `Content-Disposition` tu backend.

## 9) Layout detect and domain note

- Upload PDF runtime van su dung `LayoutDetector` de tao block layout truoc khi build context.
- Khi thay/retrain model trong `layout_detect/`, can giu on dinh class mapping de tranh vo downstream parser.
- He thong duoc toi uu cho tai lieu kien truc tieng Nhat:
  - count/area DXF branch phu thuoc convention symbol/text tieng Nhat,
  - tai lieu ngoai mien nay co the can fallback vision/context va can review thu cong.

