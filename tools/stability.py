#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""개발용: 시작 중심을 일부러 틀어도 같은 각도가 나오는지 (국소 최적점 확인)"""
import sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hik"))
sys.path.insert(0, str(ROOT))
import matcher as M, common as C
cfg = C.load_config(ROOT / "config.yaml"); calib = C.load_calib(cfg); m = C.load_model(cfg)
for f in sys.argv[1:]:
    im = cv2.imread(f, 0); cam = M.Camera(calib, m, im.shape)
    vt = M.VisibilityTable(m, cam, 1.0, C.vis_cache_path(cfg, calib)); g = M.Grads(im)
    out = []
    for dx, dy in ((0, 0), (35, 0), (-30, 25), (0, -40), (-40, -35)):
        r = M.find_pose(im, m, cam, vt, cfg.get("matching"), center0=cam.c0 + [dx, dy], grads=g)
        out.append(f"{r['psi'] % 180:6.2f}°")
    print(Path(f).name, "모델 회전각(180° 주기):", " ".join(out))
