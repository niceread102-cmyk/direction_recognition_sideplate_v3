#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공통: 설정/보정 파일, STL 모델 캐시, 각도 계산, 한글 글자 그리기"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "cache"                                      # STL 분석·가림 표 캐시 (상위 폴더)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                           # stl_model 은 상위 폴더


# ============================================================ 각도
def wrap180(a: float) -> float:
    """(-180, 180] 로"""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def plane_from_model(model_deg: float, psi: float) -> float:
    """모델 각도(반시계 +) → 화면 평면 각도(위에서 볼 때 시계 +, 3시 = 0°)"""
    return (-(model_deg + psi)) % 360.0


def user_from_plane(plane_deg: float, cfg) -> float:
    """
    화면 평면 각도(3시 = 0°) → 사용자 각도 (config angles.zero_at: top 이면 12시 = 0°, 시계 +)
    angles.frame_offset_deg: 화면과 기준(각도기/스테이지) 사이의 회전. 사용자 각도 = 화면 각도 + 이 값
                             (Photoneo 는 보정 파일의 frame_offset_deg 를 서버가 넣는다. 기본 0)
    """
    a = cfg.get("angles", {})
    z = a.get("zero_at", "top")
    off = {"top": 90.0, "right": 0.0}.get(z)
    if off is None:
        raise ValueError(f"angles.zero_at 는 top | right : {z}")
    return (plane_deg + off + float(a.get("frame_offset_deg", 0.0))) % 360.0


def plane_from_user(user_deg: float, cfg) -> float:
    a = cfg.get("angles", {})
    z = a.get("zero_at", "top")
    return (user_deg - {"top": 90.0, "right": 0.0}[z] - float(a.get("frame_offset_deg", 0.0))) % 360.0


# ============================================================ 설정 / 보정
def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = str(path.parent.resolve())
    check_config(cfg)
    return cfg


def check_config(cfg):
    t = cfg.get("target", {})
    side = str(t.get("pattern_side", "left")).lower()
    if side not in ("left", "right"):
        raise ValueError(f"target.pattern_side 는 left | right : {side}")
    feat = cfg.get("decision", {}).get("feature", "line_ratio")
    if feat not in ("rgb_sum", "edge_density", "line_ratio", "edge_ratio"):
        raise ValueError(f"decision.feature 는 rgb_sum | edge_density | line_ratio | edge_ratio : {feat}")
    reg = cfg.get("decision", {}).get("region", "half")
    if reg not in ("half", "pattern_zone"):
        raise ValueError(f"decision.region 은 half | pattern_zone : {reg}")


def rel_path(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_dir"]) / p


def calib_path(cfg):
    return rel_path(cfg, cfg["calibration"]["file"])


def load_calib(cfg):
    p = calib_path(cfg)
    if not p.exists():
        raise SystemExit(f"보정 파일이 없습니다: {p}\n먼저 python3 calibrate.py click --camera 로 보정하세요.")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg):
    from stl_model import PartModel
    s = cfg.get("stl", {})
    return PartModel(rel_path(cfg, s.get("file", "SIDE-PLATE-BP.stl")), s, cache_dir=CACHE)


def vis_cache_path(cfg, calib):
    """보이는 모서리 표 캐시 파일 (보정 타원 모양 + STL 이 같으면 재사용)"""
    key = json.dumps([round(float(calib[k]), 2) for k in ("a", "b", "phi_deg")] +
                     [str(cfg.get("stl", {}))], ensure_ascii=False)
    return CACHE / f"vis_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"


# ============================================================ 한글 글자
_FONTS = {}
_FONT_FILES = {
    False: ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
           "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
           "C:/Windows/Fonts/malgunbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}


def font(px, bold=False):
    key = (int(px), bool(bold))
    if key not in _FONTS:
        f = None
        try:
            from PIL import ImageFont
            for n in _FONT_FILES[bool(bold)]:
                if Path(n).exists():
                    f = ImageFont.truetype(n, int(px))
                    break
        except Exception:
            f = None
        _FONTS[key] = f
    return _FONTS[key]


def put_text_kr(img_bgr, text, xy, px, rgb=(255, 255, 255), bold=False, bg=None):
    """한글 글자 (PIL + 나눔고딕). PIL/글꼴이 없으면 cv2 로 (한글은 ? 로 보임)"""
    f = font(px, bold)
    if f is None:
        cv2.putText(img_bgr, text, (int(xy[0]), int(xy[1] + px)), cv2.FONT_HERSHEY_SIMPLEX, px / 30.0,
                    rgb[::-1], 1, cv2.LINE_AA)
        return img_bgr
    from PIL import Image, ImageDraw
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    if bg is not None:
        l, t, r, b = d.textbbox(xy, text, font=f)
        d.rectangle((l - 6, t - 4, r + 6, b + 4), fill=bg)
    d.text(xy, text, font=f, fill=rgb)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
