#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL 모서리 ↔ 영상 밝기 경계 맞추기 (부품의 회전각과 중심 찾기)

카메라 모델 (약투시 = 멀리서 본 기울어진 정사영)
  보정 타원(calib.yaml: cx, cy, a, b, phi_deg) = 원판 윗면 테두리(반지름 r_outer, 높이 z_rim)의 화면 모양.
  타원으로 '평면 → 화면' 2x2 변환 A 를 만들고, 높이 방향(z)이 화면에서 어느 쪽으로 몇 px 보이는지는
  정사영 조건(회전행렬의 두 행은 서로 수직인 단위벡터)으로 A 에서 계산한다. 따로 재지 않아도 된다.

좌표계
  평면 좌표 (ux, uy) : 화면처럼 오른쪽 +x, 아래(카메라 쪽) +y. 단위 = 원판 반지름. 각도 atan2(uy, ux) 는 '위에서 볼 때 시계방향 +'
  모델 좌표          : STL (위에서 볼 때 반시계 +). 모델을 psi 만큼 반시계로 돌린 뒤 (x, -y) 로 평면 좌표에 놓는다.
  → 모델 각도 m 인 방향의 평면(시계 +) 각도 = -(m + psi)
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# ============================================================ 카메라
class Camera:
    def __init__(self, calib, model, img_shape):
        h, w = img_shape[:2]
        sz = calib.get("image_size")
        s = 1.0 if not sz else w / float(sz[0])
        if sz and abs(h / float(sz[1]) - s) > 1e-3:
            raise ValueError(f"영상 크기 {w}x{h} 가 보정 때 {sz} 와 가로세로 비율이 달라 쓸 수 없습니다. 다시 보정하세요.")
        a, b = float(calib["a"]) * s, float(calib["b"]) * s
        phi = math.radians(float(calib["phi_deg"]))
        R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        self.A = R @ np.diag([a, b]) @ R.T                  # 평면(반지름 1) → 화면 px (대칭: 회전 없이 늘리기만)
        self.c0 = np.array([(float(calib["cx"]) + 0.5) * s - 0.5, (float(calib["cy"]) + 0.5) * s - 0.5])
        self.scale = s
        sc = max(a, b)
        B = self.A / sc
        r13 = math.sqrt(max(0.0, 1.0 - float(B[0] @ B[0])))
        r23 = math.sqrt(max(0.0, 1.0 - float(B[1] @ B[1])))
        if float(B[0] @ B[1]) > 0:                          # 두 행이 수직이 되도록 부호: r11 r21 + r12 r22 + r13 r23 = 0
            r23 = -r23
        zc = sc * np.array([r13, r23])
        if zc[1] > 0:                                       # 위(+z)는 화면 위쪽(-y)으로 보인다 (카메라가 앞쪽 위에 있음)
            zc = -zc
        self.zc = zc                                        # 높이 1(반지름 단위)이 화면에서 움직이는 px
        # 카메라 쪽 방향 (평면 좌표 ux, uy, up): 화면에 점으로 찍히는 방향 = 두 행에 수직
        r1 = np.array([B[0, 0], B[0, 1], zc[0] / sc])
        r2 = np.array([B[1, 0], B[1, 1], zc[1] / sc])
        v = np.cross(r1, r2)
        self.view = v / np.linalg.norm(v) * (1 if v[2] > 0 else -1)
        self.elevation_deg = math.degrees(math.asin(min(1.0, self.view[2])))
        self.R_unit = float(model.r_outer)
        self.z_ref = float(model.z_rim)
        self.Ainv = np.linalg.inv(self.A)
        # 원근: 카메라까지 거리(원판 반지름 단위). 카메라에 가까운 점일수록 크게 보인다. null 이면 약투시
        d = calib.get("persp_dist")
        self.persp = None if d in (None, 0) else float(d)

    # 모델 점 → 화면
    def project(self, p, psi_deg, center, k=1.0):
        """p: (n,3) 모델 mm. center: 화면 px (원판 윗면 테두리 높이의 중심). k: 크기 배율"""
        c, s = math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))
        x = (c * p[:, 0] - s * p[:, 1]) / self.R_unit
        y = -(s * p[:, 0] + c * p[:, 1]) / self.R_unit
        z = (p[:, 2] - self.z_ref) / self.R_unit
        if self.persp:
            q = self.view[0] * x + self.view[1] * y + self.view[2] * z      # 카메라 쪽으로 나온 거리
            k = k * self.persp / (self.persp - q)
        u = k * (self.A[0, 0] * x + self.A[0, 1] * y + self.zc[0] * z) + center[0]
        v = k * (self.A[1, 0] * x + self.A[1, 1] * y + self.zc[1] * z) + center[1]
        return u, v

    def view_in_model(self, psi_deg):
        """카메라 쪽 방향을 모델 좌표로 (psi 만큼 돌린 모델 기준)"""
        w = np.array([self.view[0], -self.view[1], self.view[2]])   # 평면 → 돌린 모델 (y 뒤집기)
        c, s = math.cos(math.radians(-psi_deg)), math.sin(math.radians(-psi_deg))
        return np.array([c * w[0] - s * w[1], s * w[0] + c * w[1], w[2]])

    def to_plane(self, x, y, center):
        """화면 px → 평면 좌표 (윗면 테두리 높이 기준)"""
        d = self.Ainv @ np.array([x - center[0], y - center[1]])
        return float(d[0]), float(d[1])

    def plane_to_image(self, u, v, center, z_mm=None):
        z = 0.0 if z_mm is None else (z_mm - self.z_ref) / self.R_unit
        d = self.A @ np.array([u, v]) + self.zc * z
        return float(center[0] + d[0]), float(center[1] + d[1])


# ============================================================ 보이는 모서리
def visible_mask(model, cam, psi_deg, horiz_step=0.2, tol=0.15):
    """psi 자세에서 카메라에 보이는 모서리 점 (면 방향 + 높이맵 가려짐)"""
    d = cam.view_in_model(psi_deg)
    face = np.maximum(model.n1 @ d, model.n2 @ d) > 0.05
    p = model.p[face]
    dxy = math.hypot(d[0], d[1])
    if dxy < 1e-6:
        out = np.zeros(len(model.p), bool)
        out[face] = True
        return out
    dt = horiz_step / dxy
    t = np.arange(horiz_step / dxy, (model.z_max - model.z_min) / max(d[2], 1e-3) + dt, dt)
    occ = np.zeros(len(p), bool)
    for tt in t:
        q = p + tt * d
        alive = q[:, 2] < model.z_max
        if not alive.any():
            break
        hz = model.height_at(q[:, 0], q[:, 1])
        occ |= alive & (hz > q[:, 2] + tol)
    out = np.zeros(len(model.p), bool)
    out[np.where(face)[0][~occ]] = True
    return out


class VisibilityTable:
    """psi 1° 간격의 보이는 점 표 (카메라 기울기만 같으면 그대로 쓸 수 있어 캐시)"""

    def __init__(self, model, cam, step_deg=1.0, cache_path=None):
        self.step = step_deg
        n = int(round(360 / step_deg))
        key = np.round(np.r_[cam.view, cam.zc / max(abs(cam.A).max(), 1)], 4)
        if cache_path is not None:
            try:
                d = np.load(cache_path)
                if d["key"].shape == key.shape and np.allclose(d["key"], key) and d["vis"].shape == (n, len(model.p)):
                    self.vis = d["vis"]
                    return
            except Exception:
                pass
        self.vis = np.stack([visible_mask(model, cam, i * step_deg) for i in range(n)])
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, key=key, vis=self.vis)
            except Exception:
                pass

    def __call__(self, psi_deg):
        return self.vis[int(round((psi_deg % 360) / self.step)) % len(self.vis)]


# ============================================================ 영상 경계
class Gradient:
    def __init__(self, gray, sigma, down=1):
        g = gray.astype(np.float32)
        if down > 1:
            g = cv2.resize(g, None, fx=1.0 / down, fy=1.0 / down, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), sigma)
        self.gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        self.gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        self.down = down
        mag = np.hypot(self.gx, self.gy)
        self.sat = float(np.percentile(mag, 99.0)) + 1e-6   # 아주 강한 경계(반사 등)가 점수를 독차지하지 않게 상한

    def sample(self, u, v):
        """화면(원본 px) 좌표에서 gx, gy (양선형)"""
        x = (u + 0.5) / self.down - 0.5
        y = (v + 0.5) / self.down - 0.5
        h, w = self.gx.shape
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        x0, y0 = x.astype(np.int64), y.astype(np.int64)
        fx, fy = x - x0, y - y0

        def bil(M):
            return ((M[y0, x0] * (1 - fx) + M[y0, x0 + 1] * fx) * (1 - fy)
                    + (M[y0 + 1, x0] * (1 - fx) + M[y0 + 1, x0 + 1] * fx) * fy)
        return bil(self.gx), bil(self.gy)


POINT_WEIGHT_CFG = {"base": 0.05, "outer_cut": 0.92}


def point_weights(model, base=None, outer_cut=None):
    """
    각도를 재는 데 쓸모 있는 정도. 부품이 돌면 점은 원둘레 방향으로 움직이므로
    반지름 방향으로 뻗은 모서리(홀 옆면 등)는 각도에 민감하고, 원 모양 모서리(테두리)는 각도와 무관하다.
    w = base + |모서리 방향 · 반지름 방향|²   (base: 원 모양 모서리도 중심 맞추기에는 조금 쓰도록)
    바깥 테두리(반지름 > outer_cut)는 뺀다: 원근 때문에 먼 쪽 테두리가 약투시 모델과 수 px 어긋나
    긴 홀 대신 테두리에 맞추려다 각도가 틀어지는 것을 막는다.
    """
    base = POINT_WEIGHT_CFG["base"] if base is None else base
    outer_cut = POINT_WEIGHT_CFG["outer_cut"] if outer_cut is None else outer_cut
    r = np.hypot(model.p[:, 0], model.p[:, 1])
    er = model.p[:, :2] / (r[:, None] + 1e-9)
    radial = np.abs((model.t[:, :2] * er).sum(1))
    w = base + radial ** 2
    if outer_cut:
        w = np.where(r > outer_cut * model.r_outer, 0.0, w)
    return w


def edge_score(model, cam, grad, psi, center, k=1.0, vis=None, sel=None, w=None):
    """
    보이는 모서리 점에서 영상 밝기 경계가 모서리와 같은 방향으로 있는 정도 (0~1, 클수록 잘 맞음)
      점수 = cos⁴(밝기 기울기와 모서리 수직 방향 사이 각) × min(기울기 세기 / 기준, 1)  의 가중 평균
      → 방향이 맞는 경계만 높게 치고, 아주 강한 경계(반사, 배경)가 점수를 독차지하지 않는다
    """
    m = vis if vis is not None else np.ones(len(model.p), bool)
    if sel is not None:
        m = m & sel
    if w is not None:
        m = m & (w > 0)
    if m.sum() < 20:
        return 0.0
    p, t = model.p[m], model.t[m]
    ww = np.ones(len(p)) if w is None else w[m]
    u, v = cam.project(p, psi, center, k)
    u2, v2 = cam.project(p + 0.3 * t, psi, center, k)
    tx, ty = u2 - u, v2 - v
    ln = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / ln, tx / ln                              # 화면에서 모서리에 수직인 방향
    gx, gy = grad.sample(u, v)
    gm = np.hypot(gx, gy) + 1e-9
    cos = np.abs(gx * nx + gy * ny) / gm
    s = cos ** 4 * np.minimum(gm / (0.3 * grad.sat), 1.0)
    return float((s * ww).sum() / ww.sum())


# ============================================================ 전체 탐색
class Grads:
    """한 장의 영상에서 쓰는 기울기 영상 묶음 (거친 1/4, 중간 1/2, 원본)"""

    def __init__(self, gray):
        self.g4 = Gradient(gray, sigma=3.0, down=4)
        self.g2 = Gradient(gray, sigma=2.0, down=2)
        self.g1 = Gradient(gray, sigma=1.2, down=1)


def coarse_search(model, cam, vt, grads, center0, search_px=60.0, step_deg=2.0, step_px=12.0, top_n=6):
    """
    1/4 크기 영상에서 중심 격자 × 각도(0~180, 대칭) 전수 탐색.
    서로 다른(각도 6° 또는 중심 20px 이상 떨어진) 상위 top_n 후보 [(score, psi, center)] 를 돌려준다.
    """
    sym, w = ~model.is_pattern, point_weights(model)
    offs = np.arange(-search_px, search_px + 1e-6, step_px)
    allc = []
    for psi in np.arange(0, 180, step_deg):
        vis = vt(psi)
        for dx in offs:
            for dy in offs:
                c = np.array(center0, float) + [dx, dy]
                allc.append((edge_score(model, cam, grads.g4, psi, c, vis=vis, sel=sym, w=w), psi, c))
    allc.sort(key=lambda x: -x[0])
    picked = []
    for sc, psi, c in allc:
        if all(abs(((psi - q[1]) + 90) % 180 - 90) >= 6.0 or np.hypot(*(c - q[2])) >= 20.0 for q in picked):
            picked.append((sc, psi, c))
            if len(picked) >= top_n:
                break
    return picked


def refine(model, cam, vt, grads, psi, ctr, k=1.0, scale_search=0.03, fast=False):
    """각도·중심·크기를 번갈아 좁혀 가며 점수 최대화 → (psi, center, k, score)"""
    sym, w = ~model.is_pattern, point_weights(model)
    ctr = np.array(ctr, float)
    stages = [(grads.g2, 1.0, 6.0, scale_search / 2), (grads.g2, 0.5, 3.0, scale_search / 4),
              (grads.g1, 0.25, 1.5, scale_search / 8), (grads.g1, 0.1, 0.6, scale_search / 16),
              (grads.g1, 0.05, 0.3, 0.0)]
    if fast:
        stages = [(grads.g2, 0.5, 3.0, scale_search / 4), (grads.g1, 0.2, 1.0, scale_search / 10)]
    cur = 0.0
    for grad, dpsi, dpx, dk in stages:
        improved, it = True, 0
        cur = edge_score(model, cam, grad, psi, ctr, k, vt(psi), sym, w)
        while improved and it < 12:
            improved, it = False, it + 1
            trials = [(psi + s * dpsi, ctr, k) for s in (-1, 1)]
            trials += [(psi, ctr + [sx * dpx, sy * dpx], k) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if sx or sy]
            if dk > 0:
                trials += [(psi, ctr, k * (1 + s * dk)) for s in (-1, 1)]
            for tp, tc, tk in trials:
                sc = edge_score(model, cam, grad, tp, tc, tk, vt(tp), sym, w)
                if sc > cur + 1e-6:
                    cur, psi, ctr, k, improved = sc, tp, np.array(tc, float), tk, True
    return psi, ctr, k, cur


def find_pose(gray, model, cam, vt, cfg_m=None, center0=None, grads=None):
    """
    반환 dict(psi, center, k, score, peak_ratio, pattern_score)
      psi   : 모델 회전각 (위에서 볼 때 반시계, 모델 → 현재 자세)
      center: 원판 윗면 테두리 중심 (화면 px)
    형상은 180° 대칭이라 psi 와 psi+180 의 점수가 거의 같다 → 문양 판정으로 고른다.
    """
    c = dict(cfg_m or {})
    grads = grads or Grads(gray)
    c0 = np.array(center0 if center0 is not None else cam.c0, float)
    cands = coarse_search(model, cam, vt, grads, c0, float(c.get("center_search_px", 60)),
                          float(c.get("coarse_step_deg", 2.0)), top_n=int(c.get("candidates", 6)))
    ss = float(c.get("scale_search", 0.03))
    # 후보마다 빠르게 다듬고, 가장 좋은 2개만 끝까지 다듬는다 (국소 최적점에 빠지지 않게)
    quick = sorted((refine(model, cam, vt, grads, psi0, ctr0, 1.0, ss, fast=True) for _, psi0, ctr0 in cands),
                   key=lambda r: -r[3])[:2]
    psi, ctr, k, _ = max((refine(model, cam, vt, grads, q[0], q[1], q[2], ss / 2) for q in quick), key=lambda r: r[3])
    sym, w = ~model.is_pattern, point_weights(model)
    s0 = edge_score(model, cam, grads.g1, psi, ctr, k, vt(psi), sym, w)
    # 각도 신뢰도: 최고점이 다른 각도(±10° 밖, 180° 대칭 제외)보다 얼마나 높은가
    others = [edge_score(model, cam, grads.g2, psi + d, ctr, k, vt(psi + d), sym, w)
              for d in np.arange(10, 171, 4.0)]
    s0_g2 = edge_score(model, cam, grads.g2, psi, ctr, k, vt(psi), sym, w)
    peak_ratio = s0_g2 / max(max(others), 1e-9)
    # 문양 모서리만의 점수 (psi vs psi+180): STL 의 문양 위치로 본 참고 판정
    pat = model.is_pattern
    sp0 = edge_score(model, cam, grads.g1, psi, ctr, k, vt(psi), pat)
    sp1 = edge_score(model, cam, grads.g1, psi + 180, ctr, k, vt(psi + 180), pat)
    return dict(psi=psi % 360, center=ctr, k=k, score=s0, peak_ratio=peak_ratio, pattern_score=(sp0, sp1))
