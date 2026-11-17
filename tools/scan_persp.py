#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개발용: 클릭 타원은 고정하고 카메라 거리(원근)만 바꿔 가며 사진별 점수/봉우리비 비교
  python3 tools/scan_persp.py --calib tools/v1_calib.yaml images/base/rot_*.png"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import matcher as M                                         # noqa: E402
from stl_model import PartModel                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--calib", required=True)
ap.add_argument("--dists", default="0,30,20,15,12,10")
ap.add_argument("images", nargs="+")
a = ap.parse_args()
base = yaml.safe_load(open(a.calib, encoding="utf-8"))
m = PartModel(ROOT / "SIDE-PLATE-BP.stl", cache_dir=ROOT / "cache")
imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in a.images]
grads = [M.Grads(im) for im in imgs]
for d in [float(x) for x in a.dists.split(",")]:
    calib = dict(base, persp_dist=d or None)
    cam = M.Camera(calib, m, imgs[0].shape)
    vt = M.VisibilityTable(m, cam, 1.0, ROOT / "cache" / f"vis_scan_{d:g}.npz")
    rows = []
    for im, g in zip(imgs, grads):
        r = M.find_pose(im, m, cam, vt, grads=g)
        rows.append(r)
    ms = sum(r["score"] for r in rows) / len(rows)
    print(f"거리 {('없음(약투시)' if not d else f'{d:g}R'):10s} 평균점수 {ms:.4f}  " +
          "  ".join(f"[psi {r['psi']:6.1f} 점수 {r['score']:.3f} 봉우리 {r['peak_ratio']:.2f} k {r['k']:.3f}]" for r in rows))
