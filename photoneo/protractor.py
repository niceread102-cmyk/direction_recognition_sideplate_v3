#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인쇄한 각도기(tools/make_protractor.py, protractor_A4.pdf) 찾기 → 중심, 크기, 회전

각도기의 큰 검은 점 12개(30° 간격, 반지름 52 mm)를 찾아
  x_k = cx + a·sinθ_k + b·cosθ_k,   y_k = cy − a·cosθ_k + b·sinθ_k   (θ_k = 30°·k)
를 최소제곱으로 푼다(선형). 회전 = atan2(b, a), 반지름 = hypot(a, b).
  - 회전: 각도기 0° 가 화면 위쪽에서 시계방향으로 몇 도 돌아가 있나 → 각도 보정 (frame_offset_deg = −회전)
  - 중심: 부품이 각도기 중심에서 벗어나 있으면, 각도기 눈금으로 읽은 홈 각도가 실제 방향과 달라진다
          (부품 중심 3.6 mm 어긋남, 눈금 38 mm 에서 읽으면 최대 약 5°). reading() 이 그 읽힐 값을 계산한다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DOT_R_MM = 52.0                    # 큰 점 반지름 (R_TICK 35 + L10 8 + 4 + 5)
READ_R_MM = 38.0                   # 눈 읽는 곳 (눈금 35~43 mm 의 가운데쯤)


def detect(gray, min_dots=8):
    """→ dict(center, r_px, px_per_mm, rotation_deg, n_dots, rms_px, dots) 또는 None"""
    g = cv2.GaussianBlur(gray, (0, 0), 3)
    bw = (g < 0.65 * np.median(g)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    pts = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 450 < a < 2500 and 0.8 < w / h < 1.25 and a / (math.pi * (max(w, h) / 2) ** 2) > 0.85:
            pts.append(cen[i])
    pts = np.array(pts, float)
    if len(pts) < min_dots:
        return None
    # 원 위에 있는 점만 (RANSAC)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(500):
        i = rng.choice(len(pts), 3, replace=False)
        try:
            cx, cy, c = np.linalg.solve(np.c_[2 * pts[i], np.ones(3)], (pts[i] ** 2).sum(1))
        except np.linalg.LinAlgError:
            continue
        R = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
        if not (0.15 * max(gray.shape) < R < 0.6 * max(gray.shape)):       # 각도기 크기 범위만
            continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R) < 15
        if best is None or inl.sum() > best[0].sum():
            best = (inl, cx, cy)
    if best is None:
        return None
    inl, cx, cy = best
    pts = pts[inl]
    if len(pts) < min_dots:
        return None
    # 점마다 몇 번째(30°·k)인지 정하고, 점 = 중심 + M·(sinθ, −cosθ) 의 아핀(중심 2 + M 4)을 최소제곱으로.
    # 아핀이라 인쇄/종이가 가로세로 다르게 늘어난 것(이 각도기는 약 2.2%)까지 맞춘다. 원 모델보다 잔차 12.7 → 7.2 px
    for _ in range(2):
        ang = np.degrees(np.arctan2(pts[:, 0] - cx, -(pts[:, 1] - cy))) % 360
        th = np.radians(np.round(ang / 30.0) * 30.0)
        X = np.c_[np.ones_like(th), np.sin(th), -np.cos(th)]
        sx, *_ = np.linalg.lstsq(X, pts[:, 0], rcond=None)
        sy, *_ = np.linalg.lstsq(X, pts[:, 1], rcond=None)
        cx, cy = sx[0], sy[0]
    Mx = np.array([[sx[1], sx[2]], [sy[1], sy[2]]])
    res = np.c_[X @ sx - pts[:, 0], X @ sy - pts[:, 1]]
    U, S, Vt = np.linalg.svd(Mx)
    Rm = U @ Vt                                               # 극분해의 회전 부분
    if np.linalg.det(Rm) < 0:
        return None                                           # 뒤집힌 영상 (있을 수 없음)
    return dict(center=(float(cx), float(cy)), M=Mx.tolist(),
                r_px=float(math.sqrt(abs(np.linalg.det(Mx)))),
                px_per_mm=float(math.sqrt(abs(np.linalg.det(Mx))) / DOT_R_MM),
                rotation_deg=float(math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))),
                axis_ratio=float(S[1] / S[0]), n_dots=int(len(pts)),
                rms_px=float(np.sqrt((res ** 2).sum(1).mean())), dots=pts.tolist())


def reading(prot, part_center, image_deg, r_mm=READ_R_MM):
    """
    부품 중심에서 화면 각도 image_deg(12시=0, 시계 +) 방향으로 뻗은 선이 각도기 눈금(반지름 r_mm)과 만나는 곳의
    각도기 눈금 값 (각도기 아핀으로 계산하므로 종이 회전, 찌그러짐, 부품이 중심에서 벗어난 것이 모두 들어간다).
    """
    Minv = np.linalg.inv(np.array(prot["M"]))
    q0 = Minv @ (np.asarray(part_center, float) - np.array(prot["center"]))   # 각도기 좌표 (점 반지름 = 1)
    t = math.radians(image_deg)
    du = Minv @ np.array([math.sin(t), -math.cos(t)])
    rr = r_mm / DOT_R_MM
    a, b, c = float(du @ du), 2 * float(q0 @ du), float(q0 @ q0) - rr * rr
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("nan")
    v = q0 + (-b + math.sqrt(disc)) / (2 * a) * du
    return math.degrees(math.atan2(v[0], -v[1])) % 360
