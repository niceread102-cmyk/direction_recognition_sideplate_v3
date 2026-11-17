#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
문양 쪽 판정: 빨간 홈(바닥 홈) 두 개를 잇는 선으로 원판을 반으로 나눠 두 쪽의 밝기(RGB)를 비교한다.

  - 형상은 180° 대칭이라 두 반쪽은 문양만 빼고 똑같다 → 차이는 문양에서 나온다.
  - 비교는 '모델 좌표의 윗면 격자점'에서 한다: 윗면(평평한 면) 위에 0.25 mm 간격 점을 찍고,
    카메라 모델로 화면 위치를 구해 그 픽셀 값을 읽는다. 홀/홈/허브/테두리 단차는 높이맵으로 자동 제외되고,
    가운데 허브에 가려 안 보이는 점도 높이맵 광선 추적으로 제외된다.
  - 두 반쪽은 서로 180° 돌린 자리끼리 짝지어 같은 점만 비교한다 (한쪽이 가려진 점은 짝까지 뺀다)
    → 허브 때문에 위쪽이 가려져도 공정하게 비교된다.

특징값 (config decision.feature)
  rgb_sum      : 픽셀 B+G+R 평균 (흑백 카메라면 밝기 x 3). 각인 문양이 있는 쪽이 어둡다 (기본)
  edge_density : 밝기 기울기 평균. 문양 쪽이 선이 많아 크다
"""
from __future__ import annotations

import math

import cv2
import numpy as np


class TopSamples:
    """윗면 비교용 모델 격자점 (모델 좌표 mm)"""

    def __init__(self, model, radius_band=(0.55, 0.95), step=0.25, z_band=(2.4, 3.8), edge_margin=0.4):
        R = model.r_outer
        g = np.arange(-R, R + 1e-9, step)
        X, Y = np.meshgrid(g, g)
        X, Y = X.ravel(), Y.ravel()
        r = np.hypot(X, Y) / R
        Z = model.height_at(X, Y)
        ok = (r >= radius_band[0]) & (r <= radius_band[1]) & (Z >= z_band[0]) & (Z <= z_band[1])
        # 높이가 바뀌는 곳(모서리) 근처는 뺀다: 주변 높이 차가 크면 제외
        if edge_margin > 0:
            for dx, dy in ((edge_margin, 0), (-edge_margin, 0), (0, edge_margin), (0, -edge_margin)):
                Zn = model.height_at(X + dx, Y + dy)
                ok &= np.abs(np.where(np.isfinite(Zn), Zn, -99) - Z) < 0.3
        # 180° 짝이 있는 점만 (형상이 대칭이라 거의 다 남는다)
        Zp = model.height_at(-X, -Y)
        ok &= np.abs(np.where(np.isfinite(Zp), Zp, -99) - Z) < 0.3
        self.p = np.stack([X[ok], Y[ok], Z[ok]], 1)
        self.ang = np.degrees(np.arctan2(self.p[:, 1], self.p[:, 0])) % 360  # 모델 각도
        # 짝 인덱스: (-x, -y) 격자점
        key = {(round(x / step), round(y / step)): i for i, (x, y) in enumerate(self.p[:, :2])}
        self.pair = np.array([key.get((round(-x / step), round(-y / step)), -1) for x, y in self.p[:, :2]])


def visible_points(model, cam, psi, p, horiz_step=0.2, tol=0.15):
    """모델 점 p 가 psi 자세에서 카메라에 보이는지 (높이맵 광선 추적)"""
    d = cam.view_in_model(psi)
    dxy = math.hypot(d[0], d[1])
    occ = np.zeros(len(p), bool)
    if dxy < 1e-6:
        return ~occ
    dt = horiz_step / dxy
    for tt in np.arange(dt, (model.z_max - p[:, 2].min()) / max(d[2], 1e-3) + dt, dt):
        q = p + tt * d
        alive = q[:, 2] < model.z_max
        if not alive.any():
            break
        occ |= alive & (model.height_at(q[:, 0], q[:, 1]) > q[:, 2] + tol)
    return ~occ


def feature_image(img_bgr, feature):
    if feature == "rgb_sum":
        return img_bgr.astype(np.float32).sum(axis=2) if img_bgr.ndim == 3 else img_bgr.astype(np.float32) * 3
    g = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if feature == "edge_density":
        g = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    if feature in ("line_ratio", "edge_ratio"):
        # 조명 세기에 덜 민감한 값: (가는 어두운 선 세기 또는 경계 세기) ÷ 주변 밝기
        g = g.astype(np.float32)
        local = cv2.GaussianBlur(g, (0, 0), 12.0) + 8.0
        if feature == "line_ratio":
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            v = cv2.morphologyEx(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.MORPH_BLACKHAT, ker)   # 각인 선 = 가는 어두운 골
        else:
            gb = cv2.GaussianBlur(g, (0, 0), 1.0)
            v = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
        return v / local
    raise ValueError(f"decision.feature 는 rgb_sum | edge_density : {feature}")


def compare_halves(fimg, model, cam, ts, psi, center, k, notch_model_deg, baseline=None, zone=None):
    """
    빨간 홈 선(모델 각도 notch_model_deg ~ +180)으로 나눈 두 반쪽의 평균 특징값.
    반쪽 P = 홈 각도에서 반시계로 0~180° (모델 기준), 반쪽 Q = 나머지.
    반환 dict(vP, vQ, nP, nQ, maskP, maskQ, uv)
    """
    vis = visible_points(model, cam, psi, ts.p)
    pair_ok = ts.pair >= 0
    both = vis & pair_ok
    both[pair_ok] &= vis[ts.pair[pair_ok]]                  # 짝도 보여야 비교에 넣는다
    u, v = cam.project(ts.p, psi, center, k)
    h, w = fimg.shape[:2]
    inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    both &= inside
    both[pair_ok] &= inside[ts.pair[pair_ok]]
    val = cv2.remap(fimg, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                    cv2.INTER_LINEAR).ravel()
    if baseline is not None:
        val = val / baseline_at(baseline, cam, ts.p, psi)
    rel = (ts.ang - notch_model_deg) % 360
    inP = rel < 180
    if zone is not None:
        # 문양 자리(모델 각도 zone=(중심, 폭))와 그 180° 짝만 비교
        zc, zw = zone
        dz = np.minimum(np.abs((ts.ang - zc + 180) % 360 - 180), np.abs((ts.ang - zc) % 360 - 180))
        both &= dz <= zw / 2
    mP, mQ = both & inP, both & ~inP
    vP = float(val[mP].mean()) if mP.any() else float("nan")
    vQ = float(val[mQ].mean()) if mQ.any() else float("nan")
    return dict(vP=vP, vQ=vQ, nP=int(mP.sum()), nQ=int(mQ.sum()), maskP=mP, maskQ=mQ, u=u, v=v)


def plane_angle_of_points(cam, p, psi):
    """모델 점의 평면(화면 기준, 시계 +, 3시 = 0) 각도"""
    a = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return (-(a + psi)) % 360


def baseline_at(baseline, cam, p, psi):
    """화면 방향별 기준선(카메라에 고정된 밝기 치우침) 값을 점마다"""
    b = np.asarray(baseline, np.float32)
    idx = (plane_angle_of_points(cam, p, psi) * len(b) / 360).astype(int) % len(b)
    return b[idx]
