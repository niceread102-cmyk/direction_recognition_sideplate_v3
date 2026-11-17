#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL 에서 측정에 필요한 형상 정보를 뽑는다. (한 번 계산해서 cache/ 에 저장)

모델 좌표계 (STL 그대로, 단위 mm)
  - 원점: 가운데 홀(허브)의 회전축.  z: 허브가 솟은 쪽이 +  (카메라가 보는 윗면)
  - 각도: 위(+z)에서 내려다볼 때 +x 에서 반시계방향 +  (일반 수학 각도)

뽑는 정보
  - 높이맵      : 위에서 본 윗면 높이 (가려짐 계산용)
  - 모서리 점    : 꺾인 모서리(이면각 > edge_min_dihedral_deg)를 따라 일정 간격으로 찍은 점 + 양쪽 면의 법선
                   → 영상의 밝기 경계와 맞춰서 부품의 회전각/중심을 찾는다
  - 문양 점      : 180° 돌렸을 때 짝이 없는 모서리 점 (= 각인 문양). 형상은 180° 대칭이고 문양만 비대칭
  - 긴 홀(관통)  : 각도 2개 (180° 간격)
  - 바닥 홈      : 바닥 테두리의 작은 홈 각도 2개 (180° 간격)

  python stl_model.py SIDE-PLATE-BP.stl          # 정보 출력 + 그림(model_topview.png)
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
from pathlib import Path

import numpy as np

CACHE_VERSION = 3


# ============================================================ STL 읽기
def load_stl(path) -> np.ndarray:
    """삼각형 배열 (N, 3, 3) float64. 바이너리/ASCII 모두"""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data[84:84 + 50 * n],
                                dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            return rec["v"].astype(np.float64)
    txt = data.decode("latin-1").split()                    # ASCII STL
    vals = [(float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3]))
            for i, tok in enumerate(txt) if tok == "vertex"]
    if not vals or len(vals) % 3:
        raise ValueError(f"STL 을 읽을 수 없음: {path}")
    return np.array(vals, np.float64).reshape(-1, 3, 3)


def _circle_fit(xy):
    """최소제곱 원 맞추기 → (cx, cy, r)"""
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    return cx, cy, math.sqrt(c[2] + cx * cx + cy * cy)


# ============================================================ 높이맵 (위에서 본 윗면 높이)
def heightmap(tris, res=0.1, half=None):
    """tris: 원점 중심 모델 좌표. 반환 (H, res, half) — H[iy, ix] = (x, y) 칸의 최고 높이, 재료 없음 = -inf
    격자 중심 좌표: x = -half + (ix + 0.5) * res"""
    if half is None:
        half = float(np.abs(tris[..., :2]).max()) + 1.0
    N = int(math.ceil(2 * half / res))
    H = np.full(N * N, -np.inf)
    P = (tris[..., :2] + half) / res - 0.5                  # 격자 좌표 (칸 중심 = 정수)
    Z = tris[..., 2]
    lo = np.floor(P.min(1)).astype(int)
    hi = np.ceil(P.max(1)).astype(int)
    span = (hi - lo + 1)
    # 크기별로 묶어서 벡터화 (대부분 작은 삼각형)
    sizes = np.maximum(span[:, 0], span[:, 1])
    for s in np.unique(sizes):
        idx = np.where(sizes == s)[0]
        if s > 400:
            continue
        gx, gy = np.meshgrid(np.arange(s), np.arange(s))
        gx, gy = gx.ravel(), gy.ravel()
        for chunk in np.array_split(idx, max(1, len(idx) * s * s // 2_000_000 + 1)):
            if len(chunk) == 0:
                continue
            X = lo[chunk, 0][:, None] + gx[None, :]
            Y = lo[chunk, 1][:, None] + gy[None, :]
            A, B, C = P[chunk, 0][:, None, :], P[chunk, 1][:, None, :], P[chunk, 2][:, None, :]
            den = (B[..., 1] - C[..., 1]) * (A[..., 0] - C[..., 0]) + (C[..., 0] - B[..., 0]) * (A[..., 1] - C[..., 1])
            ok_t = np.abs(den) > 1e-12
            den = np.where(ok_t, den, 1.0)
            l1 = ((B[..., 1] - C[..., 1]) * (X - C[..., 0]) + (C[..., 0] - B[..., 0]) * (Y - C[..., 1])) / den
            l2 = ((C[..., 1] - A[..., 1]) * (X - C[..., 0]) + (A[..., 0] - C[..., 0]) * (Y - C[..., 1])) / den
            l3 = 1 - l1 - l2
            e = -1e-6
            m = (l1 >= e) & (l2 >= e) & (l3 >= e) & ok_t & (X >= 0) & (X < N) & (Y >= 0) & (Y < N)
            z = l1 * Z[chunk, 0][:, None] + l2 * Z[chunk, 1][:, None] + l3 * Z[chunk, 2][:, None]
            np.maximum.at(H, (Y[m] * N + X[m]), z[m])
    return H.reshape(N, N), res, half


# ============================================================ 모서리 점
def crease_points(tris, min_dihedral_deg=30.0, step=0.35):
    """꺾인 모서리를 따라 step mm 간격의 점. 반환 dict(p, t, n1, n2, L)"""
    v = tris.reshape(-1, 3)
    u, inv = np.unique(v.round(4), axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    fn = np.cross(u[f[:, 1]] - u[f[:, 0]], u[f[:, 2]] - u[f[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fi = np.tile(np.arange(len(f)), 3)
    es = np.sort(e, axis=1)
    key = es[:, 0].astype(np.int64) * (len(u) + 1) + es[:, 1]
    o = np.argsort(key, kind="stable")
    key, es, fi = key[o], es[o], fi[o]
    same = np.where(key[1:] == key[:-1])[0]
    E, F1, F2 = es[same], fi[same], fi[same + 1]
    dih = np.degrees(np.arccos(np.clip((fn[F1] * fn[F2]).sum(1), -1, 1)))
    sel = dih > min_dihedral_deg
    E, F1, F2 = E[sel], F1[sel], F2[sel]
    a, b = u[E[:, 0]], u[E[:, 1]]
    L = np.linalg.norm(b - a, axis=1)
    k = np.maximum(1, np.round(L / step).astype(int))
    rep = np.repeat(np.arange(len(E)), k)
    frac = (np.concatenate([np.arange(kk) for kk in k]) + 0.5) / np.repeat(k, k)
    p = a[rep] + (b - a)[rep] * frac[:, None]
    t = (b - a)[rep] / (L[rep, None] + 1e-12)
    return dict(p=p, t=t, n1=fn[F1][rep], n2=fn[F2][rep])


# ============================================================ 단면 / 각도 특징
def outer_radius_profile(tris, z0, nbins=720):
    """z = z0 단면에서 각도별 재료의 최대 반지름 (모델 각도, nbins 칸)"""
    z = tris[..., 2]
    cross = (z.min(1) < z0) & (z.max(1) > z0)
    T = tris[cross]
    ends = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        za, zb = T[:, i, 2], T[:, j, 2]
        s = np.where((za - z0) * (zb - z0) < 0, (z0 - za) / np.where(zb != za, zb - za, 1.0), np.nan)
        ends.append(T[:, i, :2] + (T[:, j, :2] - T[:, i, :2]) * s[:, None])
    ends = np.stack(ends, 1)                                # (n, 3, 2), 교차하는 변 2개만 유효
    ok = ~np.isnan(ends[..., 0])
    keep = ok.sum(1) == 2
    seg = np.stack([e[o] for e, o in zip(ends[keep], ok[keep])])   # (m, 2, 2) 단면 선분
    # 선분을 0.05 mm 간격으로 채운다 (바깥 벽은 길쭉한 삼각형이라 끝점만 쓰면 각도 칸이 빈다)
    L = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    k = np.maximum(2, np.ceil(L / 0.05).astype(int) + 1)
    rep = np.repeat(np.arange(len(seg)), k)
    frac = np.concatenate([np.linspace(0, 1, kk) for kk in k])
    pts = seg[rep, 0] + (seg[rep, 1] - seg[rep, 0]) * frac[:, None]
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360
    r = np.hypot(pts[:, 0], pts[:, 1])
    prof = np.full(nbins, np.nan)
    idx = (ang * nbins / 360).astype(int) % nbins
    np.fmax.at(prof, idx, r)
    for _ in range(5):
        nan = np.isnan(prof)
        if not nan.any():
            break
        prof[nan] = np.fmax(np.roll(prof, 1), np.roll(prof, -1))[nan]
    return prof


def _runs(mask):
    """원형 bool 배열에서 True 구간들 [(start, length)]"""
    n = len(mask)
    if mask.all():
        return [(0, n)]
    if not mask.any():
        return []
    start = int(np.argmin(mask))                            # False 인 곳에서 시작해 원형 처리
    m = np.roll(mask, -start)
    out, i = [], 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            out.append(((i + start) % n, j - i))
            i = j
        else:
            i += 1
    return out


def circmean_deg(a, w=None):
    a = np.radians(np.asarray(a, float))
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    return math.degrees(math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) % 360


# ============================================================ 모델
class PartModel:
    def __init__(self, stl_path, cfg_stl: dict | None = None, cache_dir=None):
        c = dict(cfg_stl or {})
        self.path = Path(stl_path)
        self.min_dih = float(c.get("edge_min_dihedral_deg", 30.0))
        self.step = float(c.get("edge_step_mm", 0.35))
        self.notch_max_width = float(c.get("notch_max_width_deg", 25.0))
        self.notch_override = c.get("notch_angle_deg")      # 직접 지정 (모델 각도). None 이면 자동
        raw = self.path.read_bytes()
        h = hashlib.sha1(raw + repr((CACHE_VERSION, self.min_dih, self.step)).encode()).hexdigest()[:12]
        cache = None
        if cache_dir:
            cache = Path(cache_dir) / f"model_{self.path.stem}_{h}.npz"
            if cache.exists():
                self._load(cache)
                self._derive()
                return
        self._build()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **self._state())
        self._derive()

    # ---------------------------------------------------- 계산
    def _build(self):
        tris = load_stl(self.path)
        z = tris[..., 2]
        zmax = float(z.max())
        top = tris.reshape(-1, 3)
        top = top[top[:, 2] > zmax - 0.05]                  # 허브 윗면 원(안/밖) 꼭짓점 → 회전축
        cx, cy, _ = _circle_fit(top[:, :2])
        self.center_stl = np.array([cx, cy])
        tris = tris - np.array([cx, cy, 0.0])
        self.tris = tris
        self.H, self.res, self.half = heightmap(tris, 0.1)
        cp = crease_points(tris, self.min_dih, self.step)
        self.p, self.t, self.n1, self.n2 = cp["p"], cp["t"], cp["n1"], cp["n2"]
        self.p_dense = crease_points(tris, self.min_dih, 0.05)["p"].astype(np.float32)
        r = np.hypot(tris[..., 0], tris[..., 1])
        self.r_outer = float(r.max())
        self.z_min, self.z_max = float(z.min()), zmax
        # 바닥 근처 단면의 바깥 반지름 (바닥 홈 찾기)
        self.prof_bottom = outer_radius_profile(tris, self.z_min + 1.0)
        # 허브 바깥 반지름 (윗면 높이 근처)
        hub = tris.reshape(-1, 3)
        hub = hub[hub[:, 2] > zmax - 0.05]
        self.r_hub = float(np.hypot(hub[:, 0], hub[:, 1]).max())

    def _state(self):
        return dict(center_stl=self.center_stl, H=self.H, res=self.res, half=self.half, p=self.p, t=self.t,
                    n1=self.n1, n2=self.n2, p_dense=self.p_dense, r_outer=self.r_outer, z_min=self.z_min, z_max=self.z_max,
                    prof_bottom=self.prof_bottom, r_hub=self.r_hub)

    def _load(self, path):
        d = np.load(path)
        for k in d.files:
            v = d[k]
            setattr(self, k, v.item() if v.ndim == 0 else v)

    def _derive(self):
        H, res, half = self.H, self.res, self.half
        N = H.shape[0]
        g = -half + (np.arange(N) + 0.5) * res
        X, Y = np.meshgrid(g, g)
        R = np.hypot(X, Y)
        self.grid_x, self.grid_y, self.grid_r = X, Y, R

        # 윗면 테두리 높이: 바깥 반지름 바로 안쪽 고리의 중앙값
        ring = (R > self.r_outer - 1.2) & (R < self.r_outer - 0.4) & np.isfinite(H)
        self.z_rim = float(np.median(H[ring]))

        # 관통 홀: 재료가 없는 칸(허브 구멍 제외) → 큰 덩어리 2개
        import cv2
        hole = (~np.isfinite(H)) & (R > self.r_hub + 1.0) & (R < self.r_outer - 0.5)
        n, lab, st, ce = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
        blobs = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]
        self.slots = []
        for i in blobs:
            m = lab == i
            ang = np.degrees(np.arctan2(Y[m], X[m]))
            a0 = circmean_deg(ang)
            rel = (ang - a0 + 180) % 360 - 180
            self.slots.append(dict(angle=a0, span=(a0 + rel.min(), a0 + rel.max()),
                                   r=(float(R[m].min()), float(R[m].max())),
                                   area_mm2=float(m.sum() * res * res)))
        self.slots.sort(key=lambda s: s["angle"])
        self.slot_mask = (lab > 0) & np.isin(lab, blobs)

        # 바닥 홈: 바닥 근처 단면에서 바깥 반지름이 1 mm 이상 들어간 구간 중 좁은 것
        prof = self.prof_bottom
        nb = len(prof)
        dip = prof < self.r_outer - 1.0
        cands = []
        for s, ln in _runs(dip):
            w = ln * 360.0 / nb
            mid = (s + ln / 2.0) * 360.0 / nb % 360
            depth = float(self.r_outer - np.nanmin(np.roll(prof, -s)[:ln]))
            cands.append(dict(angle=mid, width=w, depth=depth))
        self.bottom_cuts = cands
        notch = [c for c in cands if c["width"] <= self.notch_max_width]
        self.notch_axes = []
        if self.notch_override is not None:
            a = float(self.notch_override) % 360
            self.notches = [dict(angle=a, width=None, depth=None, source="config"),
                            dict(angle=(a + 180) % 360, width=None, depth=None, source="config")]
        else:
            # 좁은 홈이 여러 쌍이면 긴 홀에 가장 가까운 쌍 (= 사진의 빨간 홈. 이 STL 은 35°/215°, 다른 쌍 168.5°/348.5°)
            def gap(c):
                return min(abs(((c["angle"] - s["angle"]) + 180) % 360 - 180) for s in self.slots) if self.slots else 0
            notch.sort(key=gap)
            pick = []
            if notch:
                a0 = notch[0]["angle"]
                pick = [c for c in notch if abs(((c["angle"] - a0) % 180 + 90) % 180 - 90) < 3.0]
            pick.sort(key=lambda c: c["angle"])
            self.notches = [dict(c, source="stl") for c in pick]
            # 좁은 홈 쌍이 둘 이상이면 어느 것이 실제 빨간 홈인지 STL 만으로 알 수 없다
            self.notch_axes = sorted({round(c["angle"] % 180.0, 1) for c in notch})

        # 문양 점: 180° 돌린 위치 근처(0.3 mm)에 모서리 점이 없으면 비대칭 = 문양
        #   비교 대상은 촘촘한(0.05 mm) 점이라 점 간격 차이로 생기는 오판이 없다
        from scipy.spatial import cKDTree
        tree = cKDTree(self.p_dense)
        q = self.p * np.array([-1, -1, 1])
        d, _ = tree.query(q, k=1)
        self.is_pattern = d > 0.3
        pp = self.p[self.is_pattern]
        if len(pp):
            ang = np.degrees(np.arctan2(pp[:, 1], pp[:, 0]))
            self.pattern_angle = circmean_deg(ang)
        else:
            self.pattern_angle = None

    # ---------------------------------------------------- 조회
    @property
    def notch_axis(self):
        """바닥 홈 축 각도 (모델 각도, 0~180)"""
        if not self.notches:
            return None
        return self.notches[0]["angle"] % 180.0

    def notch_offset_from_slot(self):
        """가장 가까운 긴 홀 기준 바닥 홈 각도 (반시계 +)"""
        if not self.notches or not self.slots:
            return None
        n = self.notches[0]["angle"]
        offs = [((n - s["angle"] + 180) % 360) - 180 for s in self.slots]
        return min(offs, key=abs)

    def height_at(self, x, y):
        """모델 xy (mm) 의 윗면 높이 (재료 없음 = -inf). 가장 가까운 칸"""
        N = self.H.shape[0]
        ix = np.clip(((x + self.half) / self.res).astype(np.int64), 0, N - 1)
        iy = np.clip(((y + self.half) / self.res).astype(np.int64), 0, N - 1)
        return self.H[iy, ix]

    def summary(self):
        s = {
            "stl": str(self.path),
            "axis_center_in_stl_mm": [round(float(v), 4) for v in self.center_stl],
            "outer_radius_mm": round(self.r_outer, 3),
            "hub_radius_mm": round(self.r_hub, 3),
            "z_range_mm": [round(self.z_min, 3), round(self.z_max, 3)],
            "rim_top_z_mm": round(self.z_rim, 3),
            "slots_deg": [round(x["angle"], 2) for x in self.slots],
            "bottom_cuts": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
                            for c in self.bottom_cuts],
            "notches_deg": [round(x["angle"], 2) for x in self.notches],
            "notch_offset_from_slot_deg": (None if self.notch_offset_from_slot() is None
                                           else round(self.notch_offset_from_slot(), 2)),
            "edge_points": int(len(self.p)),
            "pattern_points": int(self.is_pattern.sum()),
            "pattern_center_deg": None if self.pattern_angle is None else round(self.pattern_angle, 2),
        }
        return s


def draw_topview(model, path, px_per_mm=12):
    import cv2
    H = model.H[::-1]                                     # 높이맵 행 0 = y 최소 → 그림은 y 위로
    Hn = np.where(np.isfinite(H), (H - model.z_min) / (model.z_max - model.z_min) * 255, 0).clip(0, 255)
    vis = cv2.applyColorMap(Hn.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis[~np.isfinite(H)] = 0
    s = px_per_mm * model.res
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    N = vis.shape[0]

    def P(x, y):                                         # 모델 xy → 그림 (y 위로)
        return int(round((x + model.half) * px_per_mm)), int(round(N - (y + model.half) * px_per_mm))

    for p, pat in zip(model.p, model.is_pattern):
        cv2.circle(vis, P(p[0], p[1]), 1, (60, 180, 255) if pat else (255, 255, 255), -1)
    for sl in model.slots:
        a = math.radians(sl["angle"])
        cv2.line(vis, P(0, 0), P(29 * math.cos(a), 29 * math.sin(a)), (255, 80, 40), 2, cv2.LINE_AA)
    for nt in model.notches:
        a = math.radians(nt["angle"])
        cv2.line(vis, P(0, 0), P(29 * math.cos(a), 29 * math.sin(a)), (40, 40, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "blue: slots  red: notches  orange: pattern edges  (model frame, y up)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        sys.exit("사용법: python stl_model.py <part.stl> [out.png]")
    m = PartModel(sys.argv[1], cache_dir=Path(__file__).resolve().parent / "cache")
    print(json.dumps(m.summary(), ensure_ascii=False, indent=2))
    out = sys.argv[2] if len(sys.argv) > 2 else "model_topview.png"
    draw_topview(m, out)
    print(f"그림: {out}")
