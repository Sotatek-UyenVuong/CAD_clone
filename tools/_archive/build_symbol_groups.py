"""build_symbol_groups.py
Phân nhóm tất cả symbol trong symbols.json thành 20 category.

Output:
  symbol_db/symbol_groups.json   – group → {description_ja, keywords, labels[]}
  symbol_db/label_to_group.json  – label → group  (reverse index)
  symbol_db/symbols_enriched.json – bản gốc + thêm field "group" + "keywords_ja"
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SYM_DB  = ROOT / "symbol_db" / "symbols.json"
OUT_GROUPS   = ROOT / "symbol_db" / "symbol_groups.json"
OUT_REVERSE  = ROOT / "symbol_db" / "label_to_group.json"
OUT_ENRICHED = ROOT / "symbol_db" / "symbols_enriched.json"

# ---------------------------------------------------------------------------
# Taxonomy definition
# Each rule: (group_id, match_fn)  – first match wins (priority order)
# ---------------------------------------------------------------------------

GROUPS: dict[str, dict] = {
    "apartment_unit": {
        "description_ja": "住戸タイプ（間取りブロック）",
        "description_en": "Apartment unit layout blocks",
        "keywords": ["住戸", "間取り", "タイプ", "LDK", "apt_unit"],
    },
    "door": {
        "description_ja": "ドア・扉・シャッター",
        "description_en": "Doors, shutters, gates",
        "keywords": ["ドア", "扉", "シャッター", "開口", "引戸", "折戸", "door", "gate", "shutter", "folding"],
    },
    "window": {
        "description_ja": "窓・サッシ・カーテンウォール",
        "description_en": "Windows, sashes, curtain walls",
        "keywords": ["窓", "サッシ", "ガラス", "window", "sash", "louvre", "louvered"],
    },
    "stair_ramp": {
        "description_ja": "階段・スロープ・梯子",
        "description_en": "Stairs, ramps, ladders",
        "keywords": ["階段", "スロープ", "梯子", "stair", "ramp", "ladder", "step"],
    },
    "elevator_escalator": {
        "description_ja": "エレベーター・エスカレーター",
        "description_en": "Elevators, escalators",
        "keywords": ["エレベーター", "エスカレーター", "EV", "elevator", "escalator"],
    },
    "toilet_bathroom": {
        "description_ja": "トイレ・浴室・洗面・衛生器具",
        "description_en": "Toilets, bathrooms, sanitary fixtures",
        "keywords": ["トイレ", "浴室", "洗面", "便器", "浴槽", "シャワー", "洗面台",
                     "toilet", "bathroom", "bathtub", "shower", "wash_basin", "hand_basin",
                     "urinal", "vanity", "mop_sink", "paper_holder", "grab_bar",
                     "towel_bar", "hand_dryer", "paper_towel"],
    },
    "kitchen_appliance": {
        "description_ja": "キッチン・家電・厨房機器",
        "description_en": "Kitchen fixtures and appliances",
        "keywords": ["キッチン", "台所", "冷蔵庫", "電子レンジ", "コンロ", "流し台",
                     "kitchen", "sink", "cooktop", "stove", "microwave", "refrigerator",
                     "faucet", "counter_sink", "counter_3slot"],
    },
    "furniture": {
        "description_ja": "家具・インテリア",
        "description_en": "Furniture and interior elements",
        "keywords": ["家具", "椅子", "テーブル", "ソファ", "ベッド", "棚", "ロッカー",
                     "chair", "desk", "sofa", "bed", "shelf", "locker", "cabinet",
                     "wardrobe", "bench", "table", "whiteboard", "baby_chair",
                     "baby_changing", "banana", "lounge", "seat_stadium"],
    },
    "pipe_plumbing": {
        "description_ja": "配管・排水・給水",
        "description_en": "Pipes, drainage, plumbing",
        "keywords": ["配管", "排水", "給水", "パイプ", "ダクト",
                     "pipe", "drain", "plumbing", "hose", "gutter_channel",
                     "floor_drain", "drainage", "rain_gutter"],
    },
    "valve": {
        "description_ja": "バルブ・弁類",
        "description_en": "Valves and fittings",
        "keywords": ["バルブ", "弁", "valve", "fitting", "flange_joint",
                     "pipe_valve", "pipe_globe_valve", "pipe_strainer"],
    },
    "structural": {
        "description_ja": "構造部材（柱・梁・鉄骨・鉄筋・基礎）",
        "description_en": "Structural members: columns, beams, steel, rebar, foundation",
        "keywords": ["柱", "梁", "鉄骨", "鉄筋", "基礎", "杭",
                     "column", "beam", "steel", "rebar", "pile", "brace",
                     "bolt", "anchor", "screw", "nail", "nut", "weld",
                     "flange", "spring", "rc_beam", "reinforcement"],
    },
    "wall_partition_lgs": {
        "description_ja": "壁・間仕切り・LGS",
        "description_en": "Walls, partitions, LGS studs",
        "keywords": ["壁", "間仕切り", "LGS", "スタッド",
                     "wall", "partition", "lgs", "mullion", "parapet",
                     "cover_panel", "wall_break"],
    },
    "electrical_lighting": {
        "description_ja": "電気・照明・スイッチ・コンセント・通信",
        "description_en": "Electrical, lighting, switches, outlets, communication",
        "keywords": ["電気", "照明", "スイッチ", "コンセント", "電話", "アンテナ", "カメラ",
                     "electrical", "light", "switch", "outlet", "panel",
                     "antenna", "cctv", "phone", "intercom", "terminal_block",
                     "plc", "generator", "transformer", "pull_box",
                     "motion_sensor", "floodlight", "spotlight"],
    },
    "fire_safety": {
        "description_ja": "防火・消火・避難設備",
        "description_en": "Fire protection, suppression, evacuation",
        "keywords": ["消火", "防火", "避難", "スプリンクラー", "感知器", "消火器",
                     "fire", "sprinkler", "emergency_exit", "exit_light",
                     "AED", "defibrillator", "smoke_vent", "rescue_chute",
                     "fire_extinguisher", "fire_alarm", "fire_detector",
                     "fire_stop", "fire_service"],
    },
    "hvac_ventilation": {
        "description_ja": "空調・換気・給排気",
        "description_en": "HVAC, ventilation, air conditioning",
        "keywords": ["空調", "換気", "給気", "排気", "エアコン", "ファン",
                     "air_conditioner", "ac_indoor", "ventilation", "fan",
                     "heater", "boiler", "water_heater", "pump",
                     "motor_actuator", "valve_actuator", "louvre"],
    },
    "accessibility": {
        "description_ja": "バリアフリー・車椅子・誘導ブロック",
        "description_en": "Accessibility, wheelchair, tactile guidance",
        "keywords": ["車椅子", "バリアフリー", "点字", "誘導ブロック",
                     "wheelchair", "accessibility", "tactile", "accessible",
                     "signage_stairs"],
    },
    "annotation_dimension": {
        "description_ja": "寸法・注記・矢印・記号・方位",
        "description_en": "Dimensions, annotations, arrows, symbols, north arrow",
        "keywords": ["寸法", "注記", "矢印", "方位", "断面", "レベル",
                     "dimension", "section", "arrow", "level_mark", "break",
                     "revision", "grid_axis", "north_arrow", "compass",
                     "slope_arrow", "centerline", "leader", "annotation",
                     "section_mark", "section_line", "section_circle",
                     "symbol_annotation", "cons_symbol", "mark",
                     "cross_X_mark", "cross_mark", "cross_plus",
                     "triangle_mark", "diamond_pattern", "target",
                     "survey_crosshair", "slit_mark"],
    },
    "hatch_pattern": {
        "description_ja": "ハッチング・塗り潰し・素材パターン",
        "description_en": "Hatch patterns, fill, material textures",
        "keywords": ["ハッチング", "塗り", "パターン", "素材",
                     "hatch", "glass_hatch", "wire_mesh", "corrugated",
                     "insulation", "stainless"],
    },
    "floor_plan_layout": {
        "description_ja": "図面枠・レイアウト・凡例",
        "description_en": "Drawing frame, layout blocks, legend",
        "keywords": ["図面枠", "凡例", "レイアウト",
                     "drawing_frame", "floor_plan", "floor_zone",
                     "legend_box", "pit_plan", "stage_plan", "stage_",
                     "rooftop_lounge", "ceiling_plan", "ceiling_grid",
                     "table_grid", "checkbox_list", "text_notes",
                     "numbered_block", "seal_label", "break_line_symbol"],
    },
    "landscape_outdoor": {
        "description_ja": "植栽・外構・車・自転車",
        "description_en": "Landscape, outdoor, vehicles, bicycle",
        "keywords": ["植栽", "外構", "樹木", "フェンス", "車", "自転車",
                     "tree", "landscape", "lawn", "fence", "road_plan",
                     "exterior", "car_", "bicycle", "street_light",
                     "manhole", "north_arrow", "arrow_north",
                     "football", "banana_fruit"],
    },
    "unknown_misc": {
        "description_ja": "未分類・自動生成ブロック",
        "description_en": "Unclassified, auto-generated blocks",
        "keywords": ["未分類", "anonymous", "auto_generated", "autocad",
                     "cad_audit", "_text_block", "empty_block",
                     "unknown_block", "undefined", "misc"],
    },
}

# ---------------------------------------------------------------------------
# Rule-based classifier (priority: first match wins)
# ---------------------------------------------------------------------------

RULES: list[tuple[str, list[str]]] = [
    # most specific first
    ("apartment_unit",    ["apt_unit_"]),
    ("elevator_escalator",["elevator", "escalator"]),
    ("stair_ramp",        ["stair", "ramp_", "ladder", "_step_"]),
    ("door",              ["door_", "folding_door", "gate_", "track_sliding_door"]),
    ("window",            ["window_", "louvre"]),
    ("fire_safety",       ["fire_", "sprinkler", "emergency_exit", "exit_light",
                           "aed_", "smoke_vent", "rescue_chute"]),
    ("accessibility",     ["wheelchair_", "tactile_", "accessibility_symbol",
                           "signage_stairs", "grab_bar_", "toilet_accessible",
                           "hand_basin_accessible", "toilet_room_mark"]),
    ("toilet_bathroom",   ["toilet_", "bathroom_", "bathtub", "shower_tray",
                           "urinal", "wash_basin", "hand_basin", "vanity_counter",
                           "mop_sink", "paper_holder", "paper_towel", "hand_dryer",
                           "towel_bar", "grab_bar", "corner_bath"]),
    ("kitchen_appliance", ["kitchen_sink", "sink_kitchen", "sink_wall",
                           "sink_counter", "counter_sink", "counter_3slot",
                           "cooktop", "microwave", "refrigerator", "faucet",
                           "cooktop_stove", "stove_gas"]),
    ("hvac_ventilation",  ["air_conditioner", "ac_indoor", "ventilation_",
                           "fan_blade", "heater_", "boiler_", "water_heater",
                           "pump_", "motor_actuator", "valve_actuator"]),
    ("electrical_lighting",["electrical_", "light_", "switch_light", "switch_",
                            "antenna", "cctv_", "phone_", "intercom",
                            "terminal_block", "plc_", "generator", "transformer",
                            "pull_box", "motion_sensor", "floodlight",
                            "spotlight_", "exit_light"]),
    ("pipe_plumbing",     ["pipe_", "drain_pipe", "drain_box", "drainage_",
                           "floor_drain", "plumbing_", "hose", "rain_gutter",
                           "gutter_channel", "pipe_manifold"]),
    ("valve",             ["valve_", "pipe_valve", "pipe_globe_valve",
                           "pipe_strainer", "pipe_check", "pipe_stop"]),
    ("structural",        ["column_", "beam_", "steel_", "rebar_", "rc_beam",
                           "pile", "brace_", "brace", "bolt_", "anchor_",
                           "screw_", "nail_", "nut", "weld_", "flange_",
                           "spring_", "reinforcement_", "structural_",
                           "ceiling_bracket", "ceiling_furring",
                           "ceiling_fixture", "hook_symbol",
                           "bracket_", "shelf_bracket", "hinge_bolt",
                           "lock_bar", "tap_keyed", "hook",
                           "truss_", "crane_hook", "chain_hoist",
                           "post_4pile", "probe_measuring",
                           "profile_U_shape", "lgs_Z_profile",
                           "lgs_INS_profile", "lgs_channel",
                           "lgs_clip", "lgs_frame"]),
    ("wall_partition_lgs",["wall_", "lgs_", "partition_", "mullion",
                           "parapet", "cover_panel", "wall_break",
                           "glass_panel_column", "glass_element"]),
    ("furniture",         ["chair", "desk_", "sofa_", "_bed", "shelf_",
                           "locker_", "locker", "cabinet", "wardrobe_",
                           "bench_", "table", "whiteboard", "baby_chair",
                           "baby_changing", "lounge_plan", "seat_stadium",
                           "storage_box"]),
    ("annotation_dimension",["dimension_", "section_", "section_mark",
                              "arrow_", "level_mark", "break_", "revision_",
                              "grid_axis_", "north_arrow", "compass_north",
                              "slope_arrow", "centerline_", "leader_",
                              "annotation_mark", "symbol_annotation",
                              "cons_symbol", "cross_", "triangle_mark",
                              "diamond_pattern", "target_", "survey_",
                              "slit_mark", "dot_", "point_", "cross_mark",
                              "sign_range", "signal_range"]),
    ("hatch_pattern",     ["hatch_", "glass_hatch", "wire_mesh",
                           "corrugated_sheet", "stainless_element"]),
    ("floor_plan_layout", ["drawing_frame", "floor_plan_", "floor_zone_",
                           "legend_box", "pit_plan", "stage_plan",
                           "stage_", "rooftop_lounge", "ceiling_plan",
                           "ceiling_grid", "table_grid", "checkbox_list",
                           "text_notes", "numbered_block", "seal_label",
                           "break_line_symbol", "break_symbol",
                           "mep", "floor_plan_block", "floor_plan_corner",
                           "floor_plan_detail", "floor_plan_grid",
                           "floor_plan_layout", "floor_plan_section",
                           "floor_plan_shop", "section_level_mark"]),
    ("landscape_outdoor", ["tree_", "landscape", "lawn", "fence_",
                           "road_plan_", "exterior_", "car_", "bicycle_",
                           "street_light_", "manhole", "ramp_slope",
                           "football_symbol", "banana_fruit",
                           "privacy_screen_"]),
    ("accessibility",     ["wheelchair", "tactile", "accessible"]),
    ("stair_ramp",        ["ramp", "step_threshold"]),
    ("unknown_misc",      ["anonymous_", "auto_generated_", "autocad_",
                           "cad_audit_", "_text_block", "empty_block",
                           "unknown_block", "undefined_", "auto_generated"]),
]


def classify(label: str) -> str:
    label_lower = label.lower()
    for group_id, patterns in RULES:
        for pat in patterns:
            if pat.lower() in label_lower:
                return group_id
    return "unknown_misc"


def main() -> None:
    with open(SYM_DB, encoding="utf-8") as f:
        syms: dict = json.load(f)

    # Deduplicated labels
    unique_labels: dict[str, str] = {}  # label → group
    for info in syms.values():
        label = info["label"]
        if label not in unique_labels:
            unique_labels[label] = classify(label)

    # Build group index
    group_index: dict[str, dict] = {}
    for gid, gmeta in GROUPS.items():
        group_index[gid] = {**gmeta, "labels": []}

    for label, gid in sorted(unique_labels.items()):
        if gid not in group_index:
            group_index["unknown_misc"]["labels"].append(label)
        else:
            group_index[gid]["labels"].append(label)

    # Sort labels within each group
    for gid in group_index:
        group_index[gid]["labels"].sort()
        group_index[gid]["count"] = len(group_index[gid]["labels"])

    # Reverse index: label → group
    label_to_group = {lbl: grp for lbl, grp in unique_labels.items()}

    # Enriched symbols: add "group" + Japanese keywords list
    enriched = {}
    for h, info in syms.items():
        label = info["label"]
        gid = unique_labels.get(label, "unknown_misc")
        enriched[h] = {
            **info,
            "group": gid,
            "group_ja": GROUPS.get(gid, {}).get("description_ja", ""),
            "keywords": GROUPS.get(gid, {}).get("keywords", []),
        }

    # Write outputs
    OUT_GROUPS.write_text(
        json.dumps(group_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUT_REVERSE.write_text(
        json.dumps(label_to_group, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUT_ENRICHED.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Written symbol_groups.json ({len(group_index)} groups)")
    print(f"Written label_to_group.json ({len(label_to_group)} unique labels)")
    print(f"Written symbols_enriched.json ({len(enriched)} entries)")
    print()
    print(f"{'Group':<25} {'Count':>5}  Description")
    print("-" * 70)
    for gid, gdata in group_index.items():
        print(f"{gid:<25} {gdata['count']:>5}  {gdata['description_ja']}")


if __name__ == "__main__":
    main()
