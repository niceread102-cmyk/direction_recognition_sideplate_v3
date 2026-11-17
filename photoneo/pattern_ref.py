#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기준 사진 비교로 문양 쪽 판정 (Photoneo 고정 카메라용)

형상은 180° 대칭이라 STL 맞추기로 찾은 회전각 psi 와 psi+180° 를 형상으로는 구별할 수 없다.
그래서 부품 윗면(고리 0.55~0.95 R)을 '모델 각도 × 반지름' 띠 영상으로 펼쳐서
  - 문양 쪽이 맞는 기준 띠 영상(pattern_ref.npz)과 그대로 비교한 닮음(정규화 상관)
  - 180° 돌려서 비교한 닮음
중 큰 쪽을 고른다. 격자/"1-4"/"25" 문양의 모양 전체를 쓰므로, 반쪽 평균값 비교(차 3~9%)보다 확실히 갈린다
(자세가 다른 사진끼리도 닮음 0.23 대 0.06).

  - 띠 영상은 STL 높이맵으로 윗면만 쓰고(홀/허브/단차 제외), 허브에 가려진 곳은 뺀다.
  - 밝기는 각인 선 크기의 무늬만 남기고(띠 통과: 4 px 흐림 − 16 px 흐림) 주변 세기로 나눈다.
    4 px 흐림이 주물 표면의 반짝임(문양과 무관한 잡음)을 지우고, 16 px 이 조명 치우침을 지운다.
    자세 3개로 '하나 빼고 시험'한 최소 닮음 차: 고역 통과(6 px) 0.075 → 띠 통과 0.165 (2026-11-12).
  - 기준은 여러 자세 사진을 모델 각도로 맞춰 평균하면 자세마다 다른 조명 영향이 더 줄어든다.

기준 만들기: python3 photoneo/calibrate_pho.py pattern-ref ...
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import decision as D

N_ANG, N_RAD, RB = 720, 64, (0.50, 0.97)
PRE = "bp2"                        # 전처리 이름 (기준 파일에 기록, 다르면 다시 만들어야 함)
BP_S1, BP_S2 = 4.0, 16.0            # 띠 통과: 4 px 흐림(주물 표면 반짝임 제거) − 16 px 흐림(조명 치우침 제거)


def unwrap(img, ctx, psi, center, k):
    """윗면 띠 영상 (행: 반지름 RB, 열: 모델 각도 0~360). 안 보이거나 윗면이 아닌 곳은 NaN"""
    m = ctx.model
    th = np.radians(np.arange(N_ANG) * 360.0 / N_ANG)
    rr = np.linspace(RB[0], RB[1], N_RAD) * m.r_outer
    T, RR = np.meshgrid(th, rr)
    x, y = (RR * np.cos(T)).ravel(), (RR * np.sin(T)).ravel()
    z = m.height_at(x, y)
    p = np.stack([x, y, np.where(np.isfinite(z), z, m.z_rim)], 1)
    vis = D.visible_points(m, ctx.cam, psi, p) & np.isfinite(z)
    u, v = ctx.cam.project(p, psi, center, k)
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g.astype(np.float32), (0, 0), BP_S1)
    hp = g1 - cv2.GaussianBlur(g1, (0, 0), BP_S2)
    hp /= np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 2 * BP_S2)) + 1.0
    val = cv2.remap(hp, u.astype(np.float32).reshape(N_RAD, N_ANG), v.astype(np.float32).reshape(N_RAD, N_ANG),
                    cv2.INTER_LINEAR).ravel()
    h, w = g.shape
    val[~vis | (u < 0) | (u >= w - 1) | (v < 0) | (v >= h - 1)] = np.nan
    return val.reshape(N_RAD, N_ANG)


def ncc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 500:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def flip180(P):
    return np.roll(P, N_ANG // 2, axis=1)


class PatternRef:
    def __init__(self, path, min_margin=0.05):
        d = np.load(path, allow_pickle=False)
        pre = str(d["pre"]) if "pre" in d else "hp6"
        if pre != PRE or d["ref"].shape != (N_RAD, N_ANG):
            raise RuntimeError(f"{Path(path).name} 는 예전 방식({pre})으로 만든 기준입니다. 다시 만드세요: "
                               f"python3 photoneo/calibrate_pho.py pattern-ref --image ...")
        self.ref = d["ref"]
        self.n = int(d["n_images"])
        self.path = Path(path)
        self.min_margin = float(min_margin)

    def decide(self, img, ctx, pose):
        """side_plate_notch.measure(decide=...) 용. psi 기준 반쪽(P/Q) 중 문양 쪽"""
        P = unwrap(img, ctx, pose["psi"], pose["center"], pose["k"])
        same, flip = ncc(self.ref, P), ncc(self.ref, flip180(P))
        if not (np.isfinite(same) and np.isfinite(flip)):
            return None                                       # 비교할 곳이 없으면 기존 반쪽 비교로
        stl_half = ctx.stl_pattern_half                       # 모델 문양 위치가 들어 있는 반쪽 (psi 가 맞을 때)
        half = stl_half if same >= flip else ("Q" if stl_half == "P" else "P")
        return dict(half=half, margin=abs(same - flip), min_margin=self.min_margin,
                    v_pattern=max(same, flip), v_plain=min(same, flip),
                    feature="기준 사진", region="reference", label="기준 사진과의 닮음",
                    same=round(same, 4), flipped=round(flip, 4), ref_images=self.n)


def build(samples):
    """
    samples: [(띠 영상 P, 문양 쪽이 맞는가 True/False/None)] 첫 장은 True/False 필수.
    None 인 장은 지금까지의 기준과 비교해 방향을 자동으로 맞춘다. → (기준, [(장 번호, 뒤집었나, 닮음)])
    """
    P0, ok0 = samples[0]
    if ok0 is None:
        raise ValueError("첫 사진은 문양 쪽이 맞는지 정해야 합니다.")
    acc = (P0 if ok0 else flip180(P0)).copy()
    cnt = np.isfinite(acc).astype(np.float32)
    acc = np.nan_to_num(acc)
    log = [(0, not ok0, None)]
    for i, (P, ok) in enumerate(samples[1:], 1):
        cur = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        if ok is None:
            s, f = ncc(cur, P), ncc(cur, flip180(P))
            ok = s >= f
            log.append((i, not ok, round(max(s, f), 3)))
        else:
            log.append((i, not ok, None))
        Q = P if ok else flip180(P)
        acc += np.nan_to_num(Q)
        cnt += np.isfinite(Q)
    ref = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return ref, log


def save(path, ref, n_images, note=""):
    np.savez_compressed(path, ref=ref, n_images=np.int32(n_images), note=np.array(note), pre=np.array(PRE),
                        n_ang=np.int32(N_ANG), n_rad=np.int32(N_RAD), rb=np.array(RB, np.float32))
