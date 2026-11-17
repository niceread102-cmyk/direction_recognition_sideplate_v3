#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개발용: 찾은 자세 주변에서 각도별 점수 (중심/크기는 각도마다 다시 맞춤)"""
import sys
from pathlib import Path
import cv2, yaml, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import matcher as M
from stl_model import PartModel
calib = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
m = PartModel(ROOT / "SIDE-PLATE-BP.stl", cache_dir=ROOT / "cache")
sym, w = ~m.is_pattern, M.point_weights(m)
for f in sys.argv[2:]:
    im = cv2.imread(f, 0); g = M.Grads(im)
    cam = M.Camera(calib, m, im.shape); vt = M.VisibilityTable(m, cam, 1.0, ROOT / "cache" / "vis_scan_0.npz")
    r = M.find_pose(im, m, cam, vt, grads=g)
    out = []
    for d in np.arange(-10, 10.1, 1.0):
        psi = r["psi"] + d
        # 이 각도로 고정하고 중심만 다시 맞춤
        best = 0
        for dx in np.arange(-9, 9.1, 3):
            for dy in np.arange(-9, 9.1, 3):
                best = max(best, M.edge_score(m, cam, g.g1, psi, r["center"] + [dx, dy], r["k"], vt(psi), sym, w))
        out.append(best)
    print(Path(f).name, f"psi={r['psi']:.1f}", " ".join(f"{v:.3f}" for v in out))
