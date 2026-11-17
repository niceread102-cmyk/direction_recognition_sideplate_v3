#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개발용: 사진들에 STL 맞추기만 돌려 보고 겹친 그림을 저장한다.
  python3 tools/try_match.py --calib calib.yaml --out tools/out 사진1.png 사진2.png ...
"""
import argparse
import sys
import time
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
ap.add_argument("--stl", default=str(ROOT / "SIDE-PLATE-BP.stl"))
ap.add_argument("--out", default=str(ROOT / "tools" / "out"))
ap.add_argument("images", nargs="+")
a = ap.parse_args()

calib = yaml.safe_load(open(a.calib, encoding="utf-8"))
m = PartModel(a.stl, cache_dir=ROOT / "cache")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cam = vt = None
for f in a.images:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if cam is None:
        cam = M.Camera(calib, m, img.shape)
        vt = M.VisibilityTable(m, cam, 1.0, ROOT / "cache" / "vis_try.npz")
    t = time.time()
    r = M.find_pose(img, m, cam, vt)
    sp = r["pattern_score"]
    print(f"{Path(f).name:18s} psi={r['psi']:7.2f}  중심=({r['center'][0]:.1f},{r['center'][1]:.1f})  "
          f"크기={r['k']:.4f}  점수={r['score']:.3f}  봉우리비={r['peak_ratio']:.3f}  "
          f"STL문양점수 {sp[0]:.3f}/{sp[1]:.3f}  {time.time() - t:.1f}s")
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vm = vt(r["psi"])
    u, v = cam.project(m.p[vm], r["psi"], r["center"], r["k"])
    for x, y, pt in zip(u, v, m.is_pattern[vm]):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 140, 255) if pt else (0, 255, 0), -1)
    c = r["center"].astype(int)
    cv2.imwrite(str(out / f"{Path(f).stem}.jpg"), vis[max(0, c[1] - 560):c[1] + 520, max(0, c[0] - 560):c[0] + 560])
