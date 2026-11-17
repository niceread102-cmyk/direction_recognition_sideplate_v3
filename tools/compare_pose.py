#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개발용: 한 사진에서 두 자세(psi, cx, cy, k)를 겹쳐 그려 비교
  python3 tools/compare_pose.py 사진 출력.jpg psi,cx,cy,k psi,cx,cy,k"""
import sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import matcher as M
import common as C
cfg = C.load_config(ROOT / "config.yaml"); calib = C.load_calib(cfg); m = C.load_model(cfg)
im = cv2.imread(sys.argv[1], 0); cam = M.Camera(calib, m, im.shape)
vt = M.VisibilityTable(m, cam, 1.0, C.vis_cache_path(cfg, calib)); g = M.Grads(im)
sym, w = ~m.is_pattern, M.point_weights(m)
tiles = []
for spec in sys.argv[3:]:
    psi, cx, cy, k = map(float, spec.split(","))
    sc = M.edge_score(m, cam, g.g1, psi, np.array([cx, cy]), k, vt(psi), sym, w)
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    vm = vt(psi) & sym
    u, v = cam.project(m.p[vm], psi, np.array([cx, cy]), k)
    for x, y in zip(u.astype(int), v.astype(int)):
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    c = (int(cx), int(cy)); t = vis[c[1] - 470:c[1] + 470, c[0] - 470:c[0] + 470].copy()
    t = C.put_text_kr(t, f"psi {psi:.1f}  점수 {sc:.3f}", (10, 10), 30, (255, 255, 0), True, (0, 0, 0))
    tiles.append(t)
cv2.imwrite(sys.argv[2], np.hstack(tiles))
