#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
side_plate 빨간 홈 방향 맞추기 - 현재 자세에서 목표 자세까지 몇 도 돌려야 하는지 계산

목표 자세 (config.yaml target)
  빨간 홈(바닥 테두리의 작은 홈) 두 개가 0°(위)와 180°(아래)에 오고,
  빨간 홈 선으로 나눈 두 반쪽 중 문양(각인 격자/숫자)이 있는 쪽이 왼쪽(또는 오른쪽)에 오도록.

흐름
  1) STL 맞추기   : STL 의 꺾인 모서리를 영상 밝기 경계에 맞춰 부품의 회전각과 중심을 찾는다 (matcher.py)
                    - 긴 홀이 허브에 가려 한쪽만 보여도, 홀 안으로 배경이 비쳐도 된다 (모서리 전체로 맞춤)
                    - 빨간 홈은 바닥에 있어 이 카메라에서는 거의 안 보인다 → STL 에서 잰 홈 위치(긴 홀에서 +12.6°)를 쓴다
  2) 문양 쪽 판정 : 빨간 홈 선으로 나눈 두 반쪽의 값(기본 line_ratio, 또는 rgb_sum 등)을 비교 (decision.py)
                    - 형상은 180° 대칭이라 두 반쪽은 문양만 다르다
                    - (참고) STL 의 문양 모서리 위치와도 맞춰 보고, 다르면 low_confidence
  3) 회전각       : 문양 쪽 가운데가 목표 방향(왼쪽 = 270°)에 오도록 → 빨간 홈은 자동으로 0°/180°

사용법
  python3 side_plate_notch.py --camera --debug debug.png         # 연결된 카메라로 1장 찍어 측정
  python3 side_plate_notch.py --camera --save images/now.png     # 찍은 사진도 저장
  python3 side_plate_notch.py --image a.png --debug debug.png    # 사진 파일로 측정
  python3 side_plate_notch.py --image a.png --json               # 결과를 JSON 으로만 (다른 프로그램 연동용)
  python3 side_plate_notch.py --image images/base/rot_*.png --fit-baseline
        # (rgb_sum 등 조명에 민감한 특징값을 쓸 때) 부품을 360° 고르게 돌려 찍은 사진으로 방향별 밝기 치우침 기준선 만들기

출력 rotate_cw_deg : 부품을 (위에서 볼 때) 시계방향으로 돌릴 각도. 음수면 반시계.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "hik"))                      # common/decision/matcher/hik 는 hik/ 에
sys.path.insert(0, str(HERE))

import common as C                                          # noqa: E402
import decision as D                                        # noqa: E402
import matcher as M                                         # noqa: E402

STATUS_KR = {
    "ok": "정상",
    "low_confidence": "문양 판정 불확실",
    "angle_uncertain": "각도 불확실",
    "not_found": "부품을 못 찾음",
}


# ============================================================ 준비
class Context:
    """한 번 만들어 두고 여러 장 측정에 재사용 (STL, 카메라 모델, 보이는 점 표)"""

    def __init__(self, cfg, calib, img_shape):
        self.cfg, self.calib = cfg, calib
        self.model = C.load_model(cfg)
        if not self.model.notches:
            raise SystemExit("STL 에서 빨간 홈을 찾지 못했습니다. config.yaml 의 stl.notch_angle_deg 를 직접 지정하세요.")
        if len(self.model.notch_axes) > 1:                          # 자동으로 고르면 반대쪽 홈을 잡을 수 있다
            raise SystemExit(f"STL 에 좁은 홈 쌍이 여러 개입니다 (축 {self.model.notch_axes}°). 실제 빨간 홈을 "
                             f"config.yaml 의 stl.notch_angle_deg 로 직접 지정하세요 (예: {self.model.notch_axes[-1]}).")
        self.cam = M.Camera(calib, self.model, img_shape)
        self.vt = M.VisibilityTable(self.model, self.cam, 1.0, C.vis_cache_path(cfg, calib))
        d = cfg.get("decision", {})
        self.ts = D.TopSamples(self.model, radius_band=tuple(d.get("radius_band", (0.55, 0.95))))
        self.notch = float(self.model.notches[0]["angle"])          # 모델 각도
        pa = self.model.pattern_angle
        self.stl_pattern_half = None if pa is None else ("P" if (pa - self.notch) % 360 < 180 else "Q")


def half_center_model(ctx, half):
    """반쪽 P(홈에서 반시계 0~180°) / Q 의 가운데 모델 각도"""
    return (ctx.notch + (90.0 if half == "P" else 270.0)) % 360


def load_baseline(cfg, calib):
    bl = calib.get("decision_baseline")
    d = cfg.get("decision", {})
    if not bl:
        return None, None
    if bl.get("feature") != d.get("feature", "line_ratio") or \
            list(bl.get("radius_band", [])) != list(d.get("radius_band", [0.55, 0.95])):
        return None, "보정 파일의 기준선이 지금 decision 설정(feature/radius_band)과 달라 쓰지 않았습니다. --fit-baseline 으로 다시 만드세요."
    return bl["sectors"], None


# ============================================================ 측정
def measure(img, ctx, want_debug=False, decide=None):
    """
    decide: (선택) 문양 쪽 판정을 바깥에서 주는 함수 decide(img, ctx, pose) → dict(half="P"|"Q", margin, min_margin,
            label, ...) 또는 None. 주면 반쪽 비교 대신 이것으로 문양 쪽을 정한다 (예: photoneo/pattern_ref.py 의 기준 사진 비교).
            None 이면 기존과 같다.
    """
    cfg, model, cam = ctx.cfg, ctx.model, ctx.cam
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc = cfg.get("matching", {})
    dc = cfg.get("decision", {})
    warns = []

    # 1) STL 맞추기
    grads = M.Grads(gray)
    pose = M.find_pose(gray, model, cam, ctx.vt, mc, grads=grads)
    psi, ctr, k = pose["psi"], pose["center"], pose["k"]
    res = {"status": "ok"}
    if pose["score"] < 0.2:
        res.update(status="not_found", message="STL 모서리가 영상과 맞지 않습니다. 부품이 화면 안(보정 위치 근처)에 있는지, "
                                               "노출과 보정(calib.yaml)을 확인하세요.",
                   match=dict(score=round(pose["score"], 4)))
        if want_debug:
            res["_debug"] = dict(pose=pose)
        return res
    if pose["peak_ratio"] < float(mc.get("min_peak_ratio", 1.08)):
        res["status"] = "angle_uncertain"
        warns.append(f"STL 맞추기 최고점이 다른 각도보다 충분히 높지 않습니다(봉우리비 {pose['peak_ratio']:.2f}). "
                     f"각도가 몇 도 틀릴 수 있습니다.")

    # 2) 문양 쪽 판정: 빨간 홈 선으로 나눈 두 반쪽 비교
    feat = dc.get("feature", "line_ratio")
    baseline, bw = load_baseline(cfg, ctx.calib)
    if bw:
        warns.append(bw)
    zone = (model.pattern_angle, float(dc.get("zone_width_deg", 90))) \
        if dc.get("region", "half") == "pattern_zone" and model.pattern_angle is not None else None
    fimg = D.feature_image(img, feat)
    cmp = D.compare_halves(fimg, model, cam, ctx.ts, psi, ctr, k, ctx.notch, baseline, zone)
    vP, vQ = cmp["vP"], cmp["vQ"]
    if not (np.isfinite(vP) and np.isfinite(vQ)):
        res.update(status="not_found", message="비교 영역이 화면 밖이거나 가려졌습니다.")
        return res
    pattern_higher = feat != "rgb_sum"                      # rgb_sum: 각인 쪽이 어둡다 / 나머지: 각인 쪽 값이 크다
    half = ("P" if vP > vQ else "Q") if pattern_higher else ("P" if vP < vQ else "Q")
    margin = abs(vP - vQ) / max(abs(vP) + abs(vQ), 1e-9)
    min_margin = float(dc.get("min_margin", 0.04))
    ext = decide(img, ctx, pose) if decide is not None else None
    if ext is not None:                                     # 바깥 판정(기준 사진 비교 등)으로 문양 쪽을 정한다
        half, margin, min_margin = ext["half"], float(ext["margin"]), float(ext["min_margin"])
    if margin < min_margin:
        if res["status"] == "ok":
            res["status"] = "low_confidence"
        warns.append(f"{ext['label']} 차이가 작아(차 {margin:.3f}) 문양 쪽 판정이 불안정합니다." if ext else
                     f"두 반쪽의 차이가 작아(차 {margin * 100:.1f}%) 문양 쪽 판정이 불안정합니다.")

    pat_plane = C.plane_from_model(half_center_model(ctx, half), psi)       # 문양 쪽 가운데 (화면 평면 각도)

    # (참고) STL 문양 모서리로 본 문양 쪽
    sp0, sp1 = pose["pattern_score"]
    stl_check = None
    if ctx.stl_pattern_half and dc.get("use_stl_pattern_check", True):
        psi_t = psi if sp0 >= sp1 else psi + 180.0
        stl_plane = C.plane_from_model(half_center_model(ctx, ctx.stl_pattern_half), psi_t)
        agree = abs(C.wrap180(stl_plane - pat_plane)) < 90.0
        stl_margin = abs(sp0 - sp1) / max(sp0 + sp1, 1e-9)
        stl_check = dict(agree=bool(agree), margin=round(stl_margin, 4))
        if not agree:
            if res["status"] == "ok":
                res["status"] = "low_confidence"
            warns.append("반쪽 비교와 STL 문양 모서리 맞추기의 판정이 서로 다릅니다. 디버그 그림에서 문양 쪽을 확인하세요.")

    # 3) 목표까지 회전
    off_user = C.user_from_plane(0.0, cfg)                  # 사용자 각도 = 평면 각도 + off
    t = cfg.get("target", {})
    axis_user = float(t.get("notch_axis_deg", 0.0))
    side = str(t.get("pattern_side", "left")).lower()
    target_pat_user = (axis_user + (-90.0 if side == "left" else 90.0)) % 360
    pat_user = (pat_plane + off_user) % 360
    delta = C.wrap180(target_pat_user - pat_user)
    if cfg.get("output", {}).get("wrap", "shortest") != "shortest":
        delta %= 360.0
    sign = float(cfg.get("output", {}).get("rotate_sign", 1))
    tol = float(cfg.get("output", {}).get("tolerance_deg", 0.5))

    notch_user = sorted([(C.plane_from_model(ctx.notch + s, psi) + off_user) % 360 for s in (0.0, 180.0)])
    slot_user = sorted([(C.plane_from_model(s_["angle"], psi) + off_user) % 360 for s_ in model.slots])
    after = sorted([round((a + delta) % 360, 3) % 360 for a in notch_user])     # 359.9996 → 360.0 으로 안 보이게

    res.update(
        angle_convention=("12시=0°, 시계방향 +" if cfg.get("angles", {}).get("zero_at", "top") == "top"
                          else "3시=0°, 시계방향 +")
        + (f", 각도기 기준 보정 {float(cfg['angles']['frame_offset_deg']):+.2f}°"
           if float(cfg.get("angles", {}).get("frame_offset_deg", 0.0)) else ""),
        rotate_cw_deg=round(delta, 3),
        stage_cmd_deg=round(delta * sign, 3),
        within_tolerance=bool(abs(C.wrap180(delta)) <= tol),
        current=dict(
            notch_deg=[round(a, 3) for a in notch_user],
            notch_axis_deg=round(notch_user[0] % 180.0, 3),
            slot_deg=[round(a, 3) for a in slot_user],
            pattern_side_center_deg=round(pat_user, 3),
        ),
        target=dict(notch_deg=[round(axis_user % 360, 3), round((axis_user + 180) % 360, 3)],
                    pattern_side=("왼쪽" if side == "left" else "오른쪽"),
                    pattern_side_center_deg=round(target_pat_user, 3)),
        after_rotation_notch_deg=[round(a, 3) for a in after],
        decision=dict(feature=feat, region=dc.get("region", "half"),
                      pattern_side_value=round(vP if half == "P" else vQ, 4),
                      plain_side_value=round(vQ if half == "P" else vP, 4),
                      margin=round(margin, 4), points=[cmp["nP"], cmp["nQ"]], baseline=baseline is not None,
                      stl_pattern_check=stl_check),
    )
    if ext is not None:                                     # 바깥 판정: 표시용 값을 그 판정의 값으로 (draw_debug/summary 가 그대로 쓴다)
        res["decision"].update(feature=ext.get("feature", "external"), region=ext.get("region", "external"),
                               pattern_side_value=round(float(ext["v_pattern"]), 4),
                               plain_side_value=round(float(ext["v_plain"]), 4),
                               halves_compare=dict(feature=feat, vP=round(vP, 4), vQ=round(vQ, 4)),
                               external={k_: v_ for k_, v_ in ext.items() if k_ not in ("half",)})
    res.update(
        match=dict(score=round(pose["score"], 4), peak_ratio=round(pose["peak_ratio"], 3),
                   center_px=[round(float(ctr[0]), 1), round(float(ctr[1]), 1)], scale=round(k, 4),
                   model_rotation_deg=round(psi % 360, 3)),
    )
    if warns:
        res["warning"] = " / ".join(warns)
    if want_debug:
        res["_debug"] = dict(pose=pose, half=half, cmp=cmp, pat_plane=pat_plane, delta=delta)
    return res


_REGION_KR = {"half": "반쪽 전체", "pattern_zone": "문양 자리", "reference": "기준 사진 비교"}


def summary_kr(res, name=None):
    """사람이 읽는 한글 요약"""
    head = f"[{name}] " if name else ""
    st = res["status"]
    if st == "not_found":
        return f"{head}상태: {STATUS_KR[st]} - {res.get('message', '')}"
    c, d, m = res["current"], res["decision"], res["match"]
    dirn = "시계방향" if res["rotate_cw_deg"] >= 0 else "반시계방향"
    lines = [
        f"{head}상태: {STATUS_KR.get(st, st)}" + ("  (목표 자세 허용 범위 안)" if res["within_tolerance"] else ""),
        f"  돌릴 각도: {res['rotate_cw_deg']:+.2f}°  ({dirn} {abs(res['rotate_cw_deg']):.2f}°, 스테이지 명령 {res['stage_cmd_deg']:+.2f}°)",
        f"  현재 빨간 홈: {c['notch_deg'][0]:.1f}°, {c['notch_deg'][1]:.1f}°   긴 홀: {c['slot_deg'][0]:.1f}°, {c['slot_deg'][1]:.1f}°"
        f"   ({res['angle_convention']})",
        f"  현재 문양 쪽 가운데: {c['pattern_side_center_deg']:.1f}°  → 목표 {res['target']['pattern_side_center_deg']:.0f}°"
        f" ({res['target']['pattern_side']})",
        f"  회전 후 빨간 홈: {res['after_rotation_notch_deg'][0]:.1f}°, {res['after_rotation_notch_deg'][1]:.1f}°",
        (f"  문양 판정(기준 사진 비교): 기준과 닮음 {d['pattern_side_value']:.3f} / 180° 뒤집으면 {d['plain_side_value']:.3f}"
         f"  (차 {d['margin']:.3f})" if d['region'] == "reference" else
         f"  문양 판정({d['feature']}, {_REGION_KR.get(d['region'], d['region'])}): 문양 쪽 {d['pattern_side_value']:.4g}"
         f" / 반대쪽 {d['plain_side_value']:.4g}  (차 {d['margin'] * 100:.1f}%)")
        + ("" if not d.get("stl_pattern_check") else
           f"   STL 문양 확인: {'일치' if d['stl_pattern_check']['agree'] else '불일치'}"),
        f"  STL 맞추기: 점수 {m['score']:.3f}, 봉우리비 {m['peak_ratio']:.2f}, 중심 ({m['center_px'][0]:.0f}, {m['center_px'][1]:.0f}) px",
    ]
    if res.get("warning"):
        lines.append(f"  경고: {res['warning']}")
    return "\n".join(lines)


# ============================================================ 디버그 그림
_C_EDGE = (90, 230, 120)
_C_NOTCH = (255, 70, 70)
_C_SLOT = (80, 140, 255)
_C_PAT = (255, 184, 64)
_C_PLAIN = (56, 200, 190)
_C_WHITE = (245, 248, 250)
_C_MUTED = (165, 178, 190)


def draw_debug(img, ctx, res):
    cfg, model, cam = ctx.cfg, ctx.model, ctx.cam
    base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dbg = res.get("_debug", {})
    pose = dbg.get("pose")
    if pose is None or res["status"] == "not_found" and "half" not in dbg:
        vis = base.copy()
        return C.put_text_kr(vis, f"{STATUS_KR['not_found']}: {res.get('message', '')}", (30, 30), 40, (255, 120, 120),
                             True, (20, 20, 30))
    psi, ctr, k = pose["psi"], pose["center"], pose["k"]
    R = model.r_outer
    bgr = lambda c: (int(c[2]), int(c[1]), int(c[0]))     # noqa: E731

    def P(mx, my, mz=None):
        u, v = cam.project(np.array([[mx, my, model.z_rim if mz is None else mz]]), psi, ctr, k)
        return int(round(u[0])), int(round(v[0]))

    def arc(r, a0, a1, z=None, n=60):
        return [P(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), z) for a in np.linspace(a0, a1, n)]

    # 잘라낼 영역 (부품 주변)
    rim = np.array(arc(R, 0, 360, n=180))
    x0, y0 = rim.min(0) - int(0.45 * R * cam.A.max() / R)
    x1, y1 = rim.max(0) + int(0.12 * R * cam.A.max() / R)
    pad = int(0.5 * max(x1 - x0, y1 - y0))
    H, W = base.shape[:2]
    x0, y0 = max(0, x0 - pad // 3), max(0, y0 - pad // 2)
    x1, y1 = min(W, x1 + pad // 3), min(H, y1 + pad // 6)
    vis = base.copy()

    # 반쪽 칠하기 (문양 쪽 = 호박색, 반대쪽 = 청록)
    half = dbg["half"]
    ov = vis.copy()
    rb = cfg.get("decision", {}).get("radius_band", [0.55, 0.95])
    for hname, a0 in (("P", ctx.notch), ("Q", ctx.notch + 180)):
        poly = np.array(arc(rb[1] * R, a0, a0 + 180) + arc(rb[0] * R, a0 + 180, a0), np.int32)
        cv2.fillPoly(ov, [poly], bgr(_C_PAT if hname == half else _C_PLAIN))
    vis = cv2.addWeighted(ov, 0.28, vis, 0.72, 0)

    # STL 모서리: 형상은 180° 대칭이라, 판정된 문양 쪽에 STL 문양(격자)이 오도록 psi / psi+180 중 골라 그린다
    psi_e = psi if (ctx.stl_pattern_half is None or half == ctx.stl_pattern_half) else psi + 180.0
    vm = ctx.vt(psi_e)
    u, v = cam.project(model.p[vm], psi_e, ctr, k)
    for x, y in zip(u.astype(int), v.astype(int)):
        if 0 <= x < W and 0 <= y < H:
            vis[y, x] = bgr(_C_EDGE)
            if x + 1 < W:
                vis[y, x + 1] = bgr(_C_EDGE)

    s = max(1.0, (x1 - x0) / 1100.0)                        # 선 굵기/글자 배율
    th = lambda v_: max(1, int(round(v_ * s)))             # noqa: E731
    # 빨간 홈 선 + 표시 (바닥 테두리 위치)
    pa, pb = P(1.12 * R * math.cos(math.radians(ctx.notch)), 1.12 * R * math.sin(math.radians(ctx.notch))), \
        P(-1.12 * R * math.cos(math.radians(ctx.notch)), -1.12 * R * math.sin(math.radians(ctx.notch)))
    cv2.line(vis, pa, pb, (20, 20, 20), th(5), cv2.LINE_AA)
    cv2.line(vis, pa, pb, bgr(_C_NOTCH), th(2.5), cv2.LINE_AA)
    labels = []
    for sgn, nm in ((0.0, "A"), (180.0, "B")):
        a = math.radians(ctx.notch + sgn)
        q = P(R * math.cos(a), R * math.sin(a), model.z_min + 2.0)
        cv2.circle(vis, q, th(16), (20, 20, 20), th(5), cv2.LINE_AA)
        cv2.circle(vis, q, th(16), bgr(_C_NOTCH), th(2.5), cv2.LINE_AA)
        ua = (C.plane_from_model(ctx.notch + sgn, psi) + C.user_from_plane(0, cfg)) % 360
        labels.append((P(1.22 * R * math.cos(a), 1.22 * R * math.sin(a)), f"빨간 홈 {nm} {ua:.1f}°", _C_NOTCH))
    # 긴 홀
    for sl in model.slots:
        a = math.radians(sl["angle"])
        rm = (sl["r"][0] + sl["r"][1]) / 2
        q = P(rm * math.cos(a), rm * math.sin(a), 2.8)
        cv2.circle(vis, q, th(9), bgr(_C_SLOT), th(2.5), cv2.LINE_AA)
    # 문양 쪽 화살표
    hc = math.radians(half_center_model(ctx, half))
    c0 = P(0, 0)
    tip = P(0.75 * R * math.cos(hc), 0.75 * R * math.sin(hc))
    cv2.arrowedLine(vis, c0, tip, (20, 20, 20), th(6), cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(vis, c0, tip, bgr(_C_PAT), th(3), cv2.LINE_AA, tipLength=0.15)
    labels.append((P(0.62 * R * math.cos(hc + math.pi / 2.6), 0.62 * R * math.sin(hc + math.pi / 2.6)),
                   "문양 쪽", _C_PAT))
    labels.append((P(0.62 * R * math.cos(hc + math.pi + math.pi / 2.6), 0.62 * R * math.sin(hc + math.pi + math.pi / 2.6)),
                   "반대쪽", _C_PLAIN))

    vis = vis[y0:y1, x0:x1].copy()
    # 글자 (잘라낸 좌표로)
    for (lx, ly), txt, col in labels:
        vis = C.put_text_kr(vis, txt, (lx - x0 - int(40 * s), ly - y0 - int(12 * s)), int(22 * s), col, True, (16, 20, 28))

    # 정보 패널
    st = res["status"]
    lines = [(f"상태: {STATUS_KR.get(st, st)}", (120, 230, 140) if st == "ok" else (255, 170, 80), 26, True)]
    if "rotate_cw_deg" in res:
        dlt = res["rotate_cw_deg"]
        lines += [
            (f"돌릴 각도 {dlt:+.2f}°", _C_WHITE, 44, True),
            (f"({'시계방향' if dlt >= 0 else '반시계방향'} {abs(dlt):.2f}°, 위에서 볼 때)", _C_MUTED, 20, False),
            (f"현재 빨간 홈 {res['current']['notch_deg'][0]:.1f}° / {res['current']['notch_deg'][1]:.1f}°", _C_NOTCH, 22, True),
            (f"목표: 빨간 홈 {res['target']['notch_deg'][0]:.0f}° / {res['target']['notch_deg'][1]:.0f}°, "
             f"문양 {res['target']['pattern_side']}", _C_MUTED, 20, False),
            (f"문양 판정 ({res['decision']['feature']}): 문양 쪽 {res['decision']['pattern_side_value']:.4g} "
             f"/ 반대쪽 {res['decision']['plain_side_value']:.4g} (차 {res['decision']['margin'] * 100:.1f}%)", _C_PAT, 20, False),
            (f"STL 맞추기 점수 {res['match']['score']:.3f}, 봉우리비 {res['match']['peak_ratio']:.2f}", _C_EDGE, 20, False),
            (f"각도: {res['angle_convention']}", _C_MUTED, 18, False),
        ]
    px, py = int(16 * s), int(14 * s)
    pw = int(560 * s)
    ph = int(sum(int(sz * 1.45) for _, _, sz, _ in lines) * s) + int(20 * s)
    ovp = vis.copy()
    cv2.rectangle(ovp, (px - int(8 * s), py - int(8 * s)), (px + pw, py + ph), (28, 22, 16), -1)
    vis = cv2.addWeighted(ovp, 0.78, vis, 0.22, 0)
    for txt, col, sz, bold in lines:
        vis = C.put_text_kr(vis, txt, (px, py), int(sz * s), col, bold)
        py += int(sz * 1.45 * s)
    if res.get("warning"):
        vis = C.put_text_kr(vis, "경고: " + res["warning"][:90], (px, vis.shape[0] - int(36 * s)), int(18 * s),
                            (255, 170, 80), False, (28, 22, 16))
    return vis


# ============================================================ 기준선 (선택)
def fit_baseline(imgs, ctx, n_sectors=36):
    """
    부품을 360° 고르게 돌려 찍은 사진들에서, 화면 방향별 특징값 평균 → 평균으로 나눈 비율.
    문양은 여러 방향에 고루 나타나 평균에서 지워지고 카메라에 고정된 치우침(조명 음영)만 남는다.
    """
    cfg, model, cam = ctx.cfg, ctx.model, ctx.cam
    feat = cfg.get("decision", {}).get("feature", "line_ratio")
    profs = []
    for name, im in imgs:
        gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        pose = M.find_pose(gray, model, cam, ctx.vt, cfg.get("matching", {}))
        if pose["score"] < 0.2:
            print(f"  {name}: 부품을 못 찾아 건너뜀")
            continue
        psi, ctr, k = pose["psi"], pose["center"], pose["k"]
        vis = D.visible_points(model, cam, psi, ctx.ts.p)
        u, v = cam.project(ctx.ts.p, psi, ctr, k)
        f = D.feature_image(im, feat)
        val = cv2.remap(f, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                        cv2.INTER_LINEAR).ravel()
        idx = (D.plane_angle_of_points(cam, ctx.ts.p, psi) * n_sectors / 360).astype(int) % n_sectors
        prof = np.full(n_sectors, np.nan)
        for i in range(n_sectors):
            sel = vis & (idx == i)
            if sel.sum() >= 30:
                prof[i] = float(val[sel].mean())
        profs.append(prof)
        print(f"  {name}: 모델 회전 {psi:.1f}°")
    if len(profs) < 2:
        raise SystemExit("부품을 찾은 사진이 2장 미만이라 기준선을 만들 수 없습니다.")
    mean = np.nanmean(np.array(profs), axis=0)
    b = mean / np.nanmean(mean)
    return np.where(np.isfinite(b), b, 1.0), len(profs)


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser(description="side_plate 빨간 홈 방향 맞추기 (돌릴 각도 계산)")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--calib", default=None, help="보정 파일 (기본: config 의 calibration.file)")
    ap.add_argument("--camera", action="store_true", help="연결된 카메라로 1장 찍어 측정")
    ap.add_argument("--image", nargs="+", default=None, help="사진 파일로 측정")
    ap.add_argument("--save", default=None, help="--camera 로 찍은 사진 저장 경로")
    ap.add_argument("--debug", default=None, help="디버그 그림 저장 경로 (여러 장이면 폴더)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로만 출력")
    ap.add_argument("--fit-baseline", action="store_true", help="방향별 밝기 치우침 기준선 만들기 (--image 여러 장)")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    if args.calib:
        cfg["calibration"]["file"] = str(Path(args.calib).resolve())
    calib = C.load_calib(cfg)

    if args.camera:
        import hik
        try:
            img = hik.grab(cfg)
        except RuntimeError as e:
            sys.exit(f"카메라 오류: {e}")
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.save, img)
            if not args.json:
                print(f"촬영 저장: {args.save}")
        imgs = [("camera", img)]
    elif args.image:
        imgs = []
        for p in args.image:
            im = cv2.imread(p)
            if im is None:
                sys.exit(f"사진을 읽을 수 없습니다: {p}")
            imgs.append((p, im))
    else:
        sys.exit("--camera 또는 --image 가 필요합니다.")

    ctx = Context(cfg, calib, imgs[0][1].shape)

    if args.fit_baseline:
        print(f"기준선 만들기: 사진 {len(imgs)}장")
        b, n = fit_baseline(imgs, ctx)
        path = C.calib_path(cfg)
        shutil.copy(path, str(path) + ".bak")
        d = cfg.get("decision", {})
        new = dict(calib, decision_baseline=dict(feature=d.get("feature", "line_ratio"),
                                                 radius_band=list(d.get("radius_band", [0.55, 0.95])),
                                                 n_images=n, sectors=[round(float(x), 4) for x in b]))
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(new, f, sort_keys=False, allow_unicode=True)
        print(f"저장: {path}  (이전 파일: {path}.bak)")
        print(f"  {len(b)}칸, 사진 {n}장 사용, 방향별 치우침 {b.min():.2f} ~ {b.max():.2f} 배")
        if n < 4:
            print("  경고: 4장 이상(0/90/180/270°, 가능하면 30° 간격) 360° 를 고르게 돌려 찍어야 문양이 평균에서 지워집니다.")
        return

    outs = []
    for i, (name, im) in enumerate(imgs):
        r = measure(im, ctx, want_debug=bool(args.debug))
        if args.debug:
            dp = Path(args.debug)
            if len(imgs) > 1:
                dp.mkdir(parents=True, exist_ok=True)
                dp = dp / (Path(name).stem + "_debug.jpg")
            else:
                dp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dp), draw_debug(im, ctx, r))
        r.pop("_debug", None)
        if not args.json:
            print(summary_kr(r, Path(name).name if len(imgs) > 1 else None))
            if args.debug:
                print(f"  디버그 그림: {dp}")
        outs.append(r)
    if args.json:
        print(json.dumps(outs[0] if len(outs) == 1 else outs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
