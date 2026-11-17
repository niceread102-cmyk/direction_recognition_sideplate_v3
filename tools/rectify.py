#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개발용: 기울어진 영상을 원판 윗면 테두리 높이 평면 기준으로 '위에서 수직으로 본' 영상으로 편다.
  python3 tools/rectify.py images/live/now_1.png [...] [--out 폴더] [--calib calib.yaml] [--radius-px 1000] [--span 1.4]
정사영(persp_dist 없음)이면 아핀, 있으면 호모그래피.  z=0(윗면 테두리 높이) 평면만 정확하다.
검증: 펴진 영상에 반지름 1 원을 그려 윗면 테두리가 겹치는지 본다."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import matcher as M                                         # noqa: E402
from stl_model import PartModel                             # noqa: E402


def rectify_matrix(cam, c0, R_px, size, k=1.0):
    """출력 픽셀 p' → 원본 픽셀 p 의 3x3. 출력: 반지름 R_px 인 정원, 중심 = 출력 영상 가운데"""
    cout = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
    Kout = np.array([[1 / R_px, 0, -cout[0] / R_px], [0, 1 / R_px, -cout[1] / R_px], [0, 0, 1]])   # p' → (u, 1)
    if cam.persp:
        d = cam.persp
        v = cam.view[:2]
        H = np.zeros((3, 3))
        H[:2, :2] = d * k * cam.A - np.outer(c0, v)
        H[:2, 2] = d * c0
        H[2, :2] = -v
        H[2, 2] = d
    else:
        H = np.eye(3)
        H[:2, :2] = k * cam.A
        H[:2, 2] = c0
    return H @ Kout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--calib", default=str(ROOT / "calib.yaml"))
    ap.add_argument("--out", default=None, help="저장 폴더 (기본: 원본 옆)")
    ap.add_argument("--radius-px", type=float, default=1000.0, help="출력에서 원판 반지름 px")
    ap.add_argument("--no-fit", action="store_true", help="STL 맞추기 없이 calib.yaml 의 중심/크기 그대로 사용")
    ap.add_argument("--search-px", type=float, default=400.0, help="calib 중심에서 부품 중심을 찾는 범위(px)")
    ap.add_argument("--center", default=None, help="x,y  calib 중심 대신 쓸 부품 중심(px). 주면 STL 맞추기를 건너뜀")
    ap.add_argument("--span", type=float, default=1.4, help="출력 영상 한 변 = 반지름 × 2 × span")
    a = ap.parse_args()
    calib = yaml.safe_load(open(a.calib, encoding="utf-8"))
    model = PartModel(ROOT / "SIDE-PLATE-BP.stl", cache_dir=ROOT / "cache")
    vt = None
    size = int(round(2 * a.radius_px * a.span))
    for f in a.images:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        cam = M.Camera(calib, model, im.shape)
        c0, k, note = cam.c0, 1.0, "calib 중심"
        if a.center:
            c0, note = np.array([float(x) for x in a.center.split(",")]), f"지정 중심 {a.center}"
        elif not a.no_fit:                                    # 영상마다 STL 로 중심/배율을 다시 찾는다 (측정과 같은 방식)
            if vt is None:
                vt = M.VisibilityTable(model, cam, 1.0, ROOT / "cache" / "vis_rectify.npz")
            g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            r = M.find_pose(g, model, cam, vt, {"center_search_px": a.search_px})
            c0, k = np.array(r["center"], float), r["k"]
            note = f"STL 맞춤 중심 ({c0[0]:.0f},{c0[1]:.0f}) k {k:.3f} 점수 {r['score']:.2f}"
        Hm = rectify_matrix(cam, c0, a.radius_px, size, k)
        out = cv2.warpPerspective(im, Hm, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT)
        chk = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out.copy()
        cv2.circle(chk, (size // 2, size // 2), int(round(a.radius_px)), (0, 0, 255), 2, cv2.LINE_AA)
        p = Path(f)
        d = Path(a.out) if a.out else p.parent
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{p.stem}_top.png"), out)
        cv2.imwrite(str(d / f"{p.stem}_top_check.jpg"), chk)
        print(f"{f} → {d / (p.stem + '_top.png')}  (고도각 {cam.elevation_deg:.1f}°, 크기 {size}px, {note})")


if __name__ == "__main__":
    main()
