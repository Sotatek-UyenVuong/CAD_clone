#!/usr/bin/env python3
"""
symbol_labeler.py – UI for labeling DXF block symbols.

Run:
    cd /mnt/data8tb/notex/uyenvuong/CAD
    streamlit run tools/symbol_labeler.py
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
SYMBOL_DB    = Path("symbol_db")
SYMBOLS_JSON = SYMBOL_DB / "symbols.json"
IMAGES_DIR   = SYMBOL_DB / "images"
CONTEXT_DIR  = SYMBOL_DB / "context"
PAGES_DIR    = SYMBOL_DB / "pages"
CONTEXT_DIR.mkdir(exist_ok=True)
PAGES_DIR.mkdir(exist_ok=True)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Symbol Labeler", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container { padding: 1rem 1.5rem; }
.stTextArea textarea { font-family: monospace; font-size: 12px; }
.hash-badge {
    background:#1e1e2e; color:#cdd6f4;
    font-family:monospace; font-size:13px;
    padding:2px 8px; border-radius:4px; display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── load / save ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_db() -> dict:
    return json.loads(SYMBOLS_JSON.read_text(encoding="utf-8"))

def save_db(db: dict) -> None:
    SYMBOLS_JSON.write_text(json.dumps(db, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    st.cache_data.clear()

def load_image(h: str) -> Image.Image | None:
    p = IMAGES_DIR / f"{h}.png"
    return Image.open(p) if p.exists() else None

def load_context(h: str) -> Image.Image | None:
    p = CONTEXT_DIR / f"{h}.png"
    return Image.open(p) if p.exists() else None

def generate_context_for(h: str) -> Image.Image | None:
    """Call gen_context.py to generate context image on demand."""
    subprocess.run(
        [sys.executable, "tools/gen_context.py", "--hash", h],
        capture_output=True, text=True, timeout=30,
    )
    return load_context(h)

def load_page_thumb(h: str, info: dict) -> Image.Image | None:
    """Find an existing page thumbnail for this hash (any DXF file)."""
    existing = sorted(PAGES_DIR.glob(f"*_{h}.png"))
    if existing:
        return Image.open(existing[0])
    return None

def generate_page_thumb_for(h: str) -> Image.Image | None:
    """Call gen_page_thumb.py to generate page thumbnail on demand."""
    subprocess.run(
        [sys.executable, "tools/gen_page_thumb.py", "--hash", h],
        capture_output=True, text=True, timeout=60,
    )
    return load_page_thumb(h, {})

def nav_to(h: str, db: dict) -> None:
    st.session_state.selected_hash = h
    st.session_state.json_text = (
        json.dumps({h: db[h]}, ensure_ascii=False, indent=2) if h in db else "")
    st.session_state.confirm_delete = False

# ── session defaults ──────────────────────────────────────────────────────────
for k, v in [("selected_hash", ""), ("json_text", ""), ("confirm_delete", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── header ────────────────────────────────────────────────────────────────────
st.title("🔍 Symbol Labeler")
db = load_db()
total   = len(db)
labeled = sum(1 for v in db.values() if v.get("label","?") not in ("?","",None))
st.caption(f"DB: **{total}** symbols  |  labeled: **{labeled}**  |  "
           f"unlabeled: **{total - labeled}**")

# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with left:
    st.subheader("Search")
    col_q, col_f = st.columns([3, 2])
    with col_q:
        query = st.text_input("Hash hoặc label", placeholder="e.g.  39486910  hoặc  toilet",
                              label_visibility="collapsed")
    with col_f:
        show_mode = st.selectbox("Hiện", ["Tất cả","Chưa label","Đã label"],
                                 label_visibility="collapsed")

    results: list[tuple[str, dict]] = []
    for h, info in db.items():
        lbl = info.get("label","?") or "?"
        if query and query.lower() not in h.lower() and query.lower() not in lbl.lower():
            continue
        if show_mode == "Chưa label" and lbl not in ("?","",None):
            continue
        if show_mode == "Đã label" and lbl in ("?","",None):
            continue
        results.append((h, info))

    results.sort(key=lambda x: (x[1].get("label","?") not in ("?",""), -x[1].get("count",0)))

    st.caption(f"{len(results)} kết quả")

    PAGE = 48
    if len(results) > PAGE:
        page_n = st.number_input("Trang", min_value=1,
                                 max_value=(len(results)-1)//PAGE+1,
                                 value=1, step=1)
        page_results = results[(page_n-1)*PAGE : page_n*PAGE]
    else:
        page_results = results

    COLS = 6
    for row in [page_results[i:i+COLS] for i in range(0, len(page_results), COLS)]:
        cols = st.columns(COLS)
        for col, (h, info) in zip(cols, row):
            lbl = info.get("label","?") or "?"
            img = load_image(h)
            with col:
                if img:
                    st.image(img, use_container_width=True,
                             caption=f"×{info.get('count',0)}")
                else:
                    st.markdown("_(no img)_")
                btn_lbl = f"✏️ {h[:8]}" if lbl == "?" else f"✅ {lbl[:8]}"
                if st.button(btn_lbl, key=f"sel_{h}", width="stretch"):
                    nav_to(h, db)
                    st.rerun()

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with right:
    st.subheader("JSON Editor")
    h = st.session_state.selected_hash

    if not h or h not in db:
        st.info("← Chọn một symbol bên trái để chỉnh sửa.")
    else:
        info = db[h]
        img  = load_image(h)
        cur_label = info.get("label","?") or "?"

        # ── Header ────────────────────────────────────────────────────────────
        hc1, hc2 = st.columns([1, 3])
        with hc1:
            if img:
                st.image(img, width=160)
        with hc2:
            st.markdown(f"<span class='hash-badge'>{h}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Label:** `{cur_label}`")
            st.markdown(f"**Count:** {info.get('count',0)}")
            st.markdown(f"**Blocks:** `{', '.join(info.get('block_names',[])[:4])}`")
            files = info.get("files",[])
            if files:
                st.markdown(f"**Files ({len(files)}):** " +
                            ", ".join(Path(f).name for f in files[:3]) +
                            ("…" if len(files) > 3 else ""))

        # ── Context + Page view ────────────────────────────────────────────────
        ctx_img  = load_context(h)
        page_img = load_page_thumb(h, info)
        has_either = ctx_img is not None or page_img is not None

        if not has_either:
            bc1, bc2, bc3 = st.columns([2, 2, 1])
            with bc1:
                if st.button("🔍 Xem trong bản vẽ (chi tiết)",
                             key=f"gen_ctx_{h}", use_container_width=True):
                    with st.spinner("Đang render context (~5s)…"):
                        ctx_img = generate_context_for(h)
                    st.rerun()
            with bc2:
                if st.button("🗺️ Xem bản vẽ gốc (toàn trang)",
                             key=f"gen_page_{h}", use_container_width=True):
                    with st.spinner("Đang render toàn trang (~15s)…"):
                        page_img = generate_page_thumb_for(h)
                    st.rerun()
        else:
            # Show generate buttons for whichever is missing
            if ctx_img is None:
                if st.button("🔍 Tạo ảnh chi tiết", key=f"gen_ctx_{h}",
                             use_container_width=True):
                    with st.spinner("Đang render (~5s)…"):
                        ctx_img = generate_context_for(h)
                    st.rerun()
            if page_img is None:
                if st.button("🗺️ Tạo ảnh bản vẽ gốc", key=f"gen_page_{h}",
                             use_container_width=True):
                    with st.spinner("Đang render (~15s)…"):
                        page_img = generate_page_thumb_for(h)
                    st.rerun()

        if ctx_img is not None or page_img is not None:
            if ctx_img and page_img:
                v1, v2 = st.columns(2)
                with v1:
                    st.image(ctx_img,
                             caption="🔍 Chi tiết (đỏ = symbol)",
                             use_container_width=True)
                with v2:
                    with st.expander("🗺️ Bản vẽ gốc — click để phóng to",
                                     expanded=True):
                        st.image(page_img,
                                 caption=f"Bản vẽ gốc · {info.get('files',[''])[0]}",
                                 use_container_width=True)
            elif ctx_img:
                st.image(ctx_img,
                         caption="🔍 Chi tiết (đỏ = symbol)",
                         use_container_width=True)
            elif page_img:
                with st.expander("🗺️ Bản vẽ gốc — click để phóng to",
                                 expanded=True):
                    st.image(page_img,
                             caption=f"Bản vẽ gốc · {info.get('files',[''])[0]}",
                             use_container_width=True)

        st.divider()

        # ── Quick label ───────────────────────────────────────────────────────
        st.markdown("**✏️ Sửa label nhanh**")
        presets = [
            "toilet_western","toilet_accessible","toilet_wall_mount",
            "urinal","hand_basin","vanity_counter","bathtub",
            "kitchen_sink","floor_drain","shower",
            "door","window","sliding_door",
            "stair","elevator","escalator",
            "column_concrete","wall_lgs",
            "car","tree","wheelchair_symbol","ignore",
        ]
        pcols = st.columns(4)
        for i, preset in enumerate(presets):
            active = (cur_label == preset)
            with pcols[i % 4]:
                if st.button(f"{'✓ ' if active else ''}{preset}",
                             key=f"pre_{preset}",
                             type="primary" if active else "secondary",
                             width="stretch"):
                    db[h]["label"] = preset
                    save_db(db)
                    nav_to(h, db)
                    st.rerun()

        new_label = st.text_input("Hoặc nhập label tùy ý",
                                  value=cur_label if cur_label != "?" else "",
                                  placeholder="nhập rồi nhấn Lưu",
                                  key="label_input")
        if st.button("💾 Lưu label", type="primary", width="stretch"):
            db[h]["label"] = new_label or "?"
            save_db(db)
            nav_to(h, db)
            st.success(f"Saved: `{h}` → `{new_label}`")
            st.rerun()

        st.divider()

        # ── Raw JSON editor ───────────────────────────────────────────────────
        st.markdown("**📄 Raw JSON** (sửa trực tiếp)")
        if not st.session_state.json_text or h not in st.session_state.json_text:
            st.session_state.json_text = json.dumps({h: info}, ensure_ascii=False, indent=2)

        edited_json = st.text_area("JSON entry", value=st.session_state.json_text,
                                   height=280, label_visibility="collapsed",
                                   key="raw_json")
        jc1, jc2 = st.columns(2)
        with jc1:
            if st.button("💾 Lưu raw JSON", width="stretch"):
                try:
                    parsed = json.loads(edited_json)
                    if not isinstance(parsed, dict):
                        st.error("JSON phải là object {hash: {...}}")
                    else:
                        for key, val in parsed.items():
                            db[key] = {**db.get(key, {}), **val}
                        save_db(db)
                        nav_to(h, db)
                        st.success("Saved!")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"JSON error: {e}")
        with jc2:
            if st.button("↺ Reset", width="stretch"):
                st.session_state.json_text = json.dumps(
                    {h: db[h]}, ensure_ascii=False, indent=2)
                st.rerun()

        st.divider()

        # ── Delete ────────────────────────────────────────────────────────────
        st.markdown("**🗑️ Xóa entry**")

        if not st.session_state.confirm_delete:
            if st.button("🗑️ Xóa hash này khỏi JSON",
                         type="secondary", width="stretch"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.warning(
                f"Xác nhận xóa `{h}` khỏi symbols.json?\n\n"
                f"label=`{cur_label}`  |  count=`{info.get('count',0)}`"
            )
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("✅ Xác nhận xóa", type="primary",
                             width="stretch"):
                    all_sorted = [x[0] for x in sorted(
                        db.items(), key=lambda x: -x[1].get("count",0))]
                    pos_del = all_sorted.index(h) if h in all_sorted else 0
                    next_h = (all_sorted[pos_del+1]
                              if pos_del+1 < len(all_sorted)
                              else (all_sorted[pos_del-1] if pos_del > 0 else ""))
                    del db[h]
                    save_db(db)
                    nav_to(next_h, db)
                    st.toast(f"Đã xóa `{h}`", icon="🗑️")
                    st.rerun()
            with dc2:
                if st.button("❌ Hủy", width="stretch"):
                    st.session_state.confirm_delete = False
                    st.rerun()

        st.divider()

        # ── Navigate ──────────────────────────────────────────────────────────
        st.markdown("**Điều hướng**")
        all_hashes = [x[0] for x in sorted(db.items(),
                       key=lambda x: -x[1].get("count",0))]
        pos = all_hashes.index(h) if h in all_hashes else 0

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            if pos > 0 and st.button("⬅ Prev", width="stretch"):
                nav_to(all_hashes[pos-1], db)
                st.rerun()
        with nc2:
            st.caption(f"{pos+1} / {len(all_hashes)}")
        with nc3:
            if pos < len(all_hashes)-1 and st.button("Next ➡", width="stretch"):
                nav_to(all_hashes[pos+1], db)
                st.rerun()
