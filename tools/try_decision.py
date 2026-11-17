#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개발용: 맞추기 + 반쪽 비교 값을 출력 (어느 특징값이 문양 쪽을 잘 가르는지 확인)
  python3 tools/try_decision.py --calib calib.yaml 사진...
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import decision as D                                        # noqa: E402
import matcher as M                                         # noqa: E402
from stl_model import PartModel                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", required=True)
ap.add_argument("--stl", default=str(ROOT / "SIDE-PLATE-BP.stl"))
ap.add_argument("images", nargs="+")
a = ap.parse_args()

calib = yaml.safe_load(open(a.calib, encoding="utf-8"))
m = PartModel(a.stl, cache_dir=ROOT / "cache")
notch = m.notches[0]["angle"]
pat_side_model = "P" if (m.pattern_angle - notch) % 360 < 180 else "Q"
print(f"STL: 빨간 홈 {notch:.1f}°, 문양 중심 {m.pattern_angle:.1f}° → STL 기준 문양 반쪽 = {pat_side_model}")
ts = D.TopSamples(m, radius_band=(0.55, 0.95))
feats = ("rgb_sum", "edge_density", "line_ratio", "edge_ratio")
zones = {"반쪽 전체": None, "문양자리±45°": (m.pattern_angle, 90.0)}
cam = vt = None
poses = []
for f in a.images:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cam is None:
        cam = M.Camera(calib, m, img.shape)
        vt = M.VisibilityTable(m, cam, 1.0, ROOT / "cache" / "vis_try.npz")
    r = M.find_pose(gray, m, cam, vt)
    sp = r["pattern_score"]
    psi = r["psi"] if sp[0] >= sp[1] else r["psi"] + 180   # 기준 사진 4장은 눈으로 확인한 정답과 같음
    poses.append((f, img, psi, r))
for zn, zone in zones.items():
    print(f"=== 비교 영역: {zn}")
    for feat in feats:
        diffs = []
        for f, img, psi, r in poses:
            c = D.compare_halves(D.feature_image(img, feat), m, cam, ts, psi, r["center"], r["k"], notch, zone=zone)
            pat, non = (c["vP"], c["vQ"]) if pat_side_model == "P" else (c["vQ"], c["vP"])
            diffs.append(100 * (pat - non) / (pat + non))
        ok = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
        print(f"  {feat:13s} 문양쪽-반대쪽 차(%): " + "  ".join(f"{d:+6.1f}" for d in diffs) + ("   ← 4장 모두 같은 부호" if ok else ""))
