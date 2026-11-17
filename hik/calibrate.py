#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 보정 (카메라를 설치하거나 옮겼을 때 1회)

1단계: 테두리 클릭 → 타원 초기값
    python3 calibrate.py click --camera               # 촬영 → images/calib_click.png 저장 → 클릭 창
    python3 calibrate.py click --image 사진.png
    - 원판 윗면 바깥 테두리를 고르게 8~12점 클릭, Enter 로 확정
    - 단축키: z(마지막 점 취소)  r(전체 초기화)  q/Esc(취소)

2단계: STL 로 다듬기 (권장)
    python3 calibrate.py refine images/base/rot_*.png
    - 부품을 여러 각도(예: 0/90/180/270°)로 돌려 찍은 사진에서 STL 모서리를 맞춰
      회전 중심(cx, cy)과 전체 크기를 고친다. (--shape: 타원 모양까지, 실험 기능)
    - 사진은 validate.py capture 로 찍으면 된다.

(--config 기본값 config.yaml, 보정 파일 = config 의 calibration.file. 기존 파일은 .bak 로 백업)
"""
from __future__ import annotations

import argparse
import math
import shutil
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                              # side_plate_notch, stl_model
sys.path.insert(0, str(HERE))

import common                                               # noqa: E402
import matcher as M                                         # noqa: E402


# ============================================================ 1단계: 클릭
def fit_ellipse(pts, image_size):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    (cx, cy), (w, h), ang = cv2.fitEllipse(pts)
    a, b = w / 2.0, h / 2.0
    phi = math.radians(ang)
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    Mx = R @ np.diag([1.0 / a, 1.0 / b]) @ R.T
    q = (Mx @ (pts.reshape(-1, 2) - [cx, cy]).T).T
    rms = float(np.sqrt(np.mean((np.hypot(q[:, 0], q[:, 1]) - 1.0) ** 2)) * (a + b) / 2.0)
    return dict(cx=float(cx), cy=float(cy), a=float(a), b=float(b), phi_deg=float(ang),
                image_size=[int(image_size[0]), int(image_size[1])], fit_rms_px=rms, n_points=int(len(pts)),
                camera_elevation_deg=float(math.degrees(math.asin(min(1.0, min(a, b) / max(a, b))))))


def click_points(img):
    scale = min(1.0, 1400.0 / max(img.shape[:2]))
    disp0 = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    pts = []

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x / scale, y / scale))

    win = "rim click (Enter=OK, z=undo, r=reset, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = disp0.copy()
        for p in pts:
            cv2.circle(disp, (int(p[0] * scale), int(p[1] * scale)), 4, (0, 0, 255), -1)
        if len(pts) >= 6:
            (cx, cy), (w, h), ang = cv2.fitEllipse(np.array(pts, np.float32).reshape(-1, 1, 2))
            cv2.ellipse(disp, ((cx * scale, cy * scale), (w * scale, h * scale), ang), (0, 255, 0), 1)
        disp = common.put_text_kr(disp, f"클릭한 점: {len(pts)}개 (6개 이상 필요)  Enter=확정  z=취소  r=초기화  q=그만",
                                  (10, 10), 22, (255, 80, 80))
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 32) and len(pts) >= 6:
            break
        if k == ord("z") and pts:
            pts.pop()
        if k == ord("r"):
            pts.clear()
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("취소했습니다.")
    cv2.destroyAllWindows()
    return pts


def save_calib(cfg, calib):
    out = common.calib_path(cfg)
    if out.exists():
        shutil.copy(out, str(out) + ".bak")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(calib, f, sort_keys=False, allow_unicode=True)
    return out


def cmd_click(args, cfg):
    tag = None
    if args.camera:
        import hik
        path = Path(args.image or (HERE / "images" / "calib_click.png"))
        path.parent.mkdir(parents=True, exist_ok=True)
        img = hik.grab(cfg, check=False)
        tag = cfg["camera"]["_id"]
        cv2.imwrite(str(path), img)
        print(f"촬영 저장: {path}  ({img.shape[1]}x{img.shape[0]})  카메라: {tag}")
    else:
        if not args.image:
            sys.exit("--image 또는 --camera 가 필요합니다.")
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"사진을 읽을 수 없습니다: {args.image}")
        print("참고: 사진 파일로 보정하면 카메라 기록이 남지 않아 측정 때 경고가 나옵니다. (--camera 권장)")
    h, w = img.shape[:2]
    if args.points:
        pts = [tuple(float(v) for v in p.split(",")) for p in args.points.split()]
    else:
        pts = click_points(img)
    if len(pts) < 6:
        sys.exit("점이 6개 이상 필요합니다.")
    calib = fit_ellipse(pts, (w, h))
    calib["source"] = "테두리 클릭"
    if tag:
        calib["camera"] = tag
    out = save_calib(cfg, calib)
    print(f"저장: {out}")
    print(f"  타원 중심 ({calib['cx']:.1f}, {calib['cy']:.1f})  반축 a={calib['a']:.1f}px b={calib['b']:.1f}px  "
          f"기울기 {calib['phi_deg']:.1f}°")
    print(f"  클릭 오차(RMS) {calib['fit_rms_px']:.2f}px   카메라 고도각 약 {calib['camera_elevation_deg']:.1f}° (정면=90°)")
    if calib["fit_rms_px"] > 3.0:
        print("  경고: 클릭 오차가 큽니다. 테두리를 더 고르게/정확히 클릭하세요.")
    print("  다음: 부품을 0/90/180/270° 로 돌려 찍고(validate.py capture) → calibrate.py refine 으로 다듬기")


# ============================================================ 2단계: STL 로 다듬기
def cmd_refine(args, cfg):
    calib = common.load_calib(cfg)
    model = common.load_model(cfg)
    imgs = []
    for p in args.images:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            sys.exit(f"사진을 읽을 수 없습니다: {p}")
        imgs.append((p, im))
    if len(imgs) < 2:
        print("경고: 2장 이상(서로 다른 각도) 써야 타원 모양이 제대로 잡힙니다.")
    cam = M.Camera(calib, model, imgs[0][1].shape)
    vt = M.VisibilityTable(model, cam, 1.0, common.vis_cache_path(cfg, calib))
    grads = [M.Grads(im) for _, im in imgs]
    mc = cfg.get("matching", {})

    print(f"사진 {len(imgs)}장에서 부품 자세를 찾는 중 ...")
    poses = []
    for (p, im), g in zip(imgs, grads):
        r = M.find_pose(im, model, cam, vt, mc, grads=g)
        poses.append([r["psi"], r["center"], r["k"]])
        print(f"  {Path(p).name:20s} 각도 {r['psi']:7.2f}°  중심 ({r['center'][0]:.1f}, {r['center'][1]:.1f})  "
              f"크기배율 {r['k']:.4f}  점수 {r['score']:.3f}")

    k_med = float(np.median([q[2] for q in poses]))
    cur = dict(calib, a=calib["a"] * k_med, b=calib["b"] * k_med)
    for q in poses:
        q[2] /= k_med

    def total(c, fast=True):
        cm = M.Camera(c, model, imgs[0][1].shape)
        s, newp = 0.0, []
        for (psi, ctr, k), g in zip(poses, grads):
            psi2, ctr2, k2, sc = M.refine(model, cm, vt, g, psi, ctr, k, scale_search=0.004, fast=fast)
            s += sc
            newp.append([psi2, ctr2, k2])
        return s / len(poses), newp

    best, poses = total(cur)
    print(f"중심/크기 맞춤 후 평균 점수 {best:.4f}")
    if args.shape:
        # (실험) 타원 모양까지 다듬기: 사진이 적으면 과적합될 수 있어 기본은 끔
        for da, dphi in ((0.015, 3.0), (0.006, 1.0), (0.002, 0.3)):
            improved, it = True, 0
            while improved and it < 8:
                improved, it = False, it + 1
                trials = []
                for s in (-1, 1):
                    trials.append(dict(cur, a=cur["a"] * (1 + s * da)))
                    trials.append(dict(cur, b=cur["b"] * (1 + s * da)))
                    trials.append(dict(cur, phi_deg=cur["phi_deg"] + s * dphi))
                for t in trials:
                    sc, newp = total(t)
                    if sc > best + 1e-5:
                        best, cur, poses, improved = sc, t, newp, True
                print(f"  모양 다듬기(크기 ±{da * 100:.1f}%, 기울기 ±{dphi}°) 반복 {it}: 점수 {best:.4f}  "
                      f"a={cur['a']:.1f} b={cur['b']:.1f} phi={cur['phi_deg']:.2f}")

    ctrs = np.array([q[1] for q in poses])
    s = M.Camera(cur, model, imgs[0][1].shape).scale
    c_med = (np.median(ctrs, axis=0) + 0.5) / s - 0.5
    new = dict(calib, cx=float(c_med[0]), cy=float(c_med[1]), a=float(cur["a"]), b=float(cur["b"]),
               phi_deg=float(cur["phi_deg"]),
               camera_elevation_deg=float(math.degrees(math.asin(min(cur["a"], cur["b"]) / max(cur["a"], cur["b"])))),
               source=f"STL 다듬기 ({len(imgs)}장, 평균 점수 {best:.4f}" + (", 모양 포함" if args.shape else ", 중심/크기") + ")")
    new.pop("fit_rms_px", None)
    new.pop("n_points", None)
    out = save_calib(cfg, new)
    print(f"저장: {out}  (이전 파일: {out}.bak)")
    print(f"  타원 a {calib['a']:.1f} → {new['a']:.1f} px,  b {calib['b']:.1f} → {new['b']:.1f} px,  "
          f"기울기 {calib['phi_deg']:.2f} → {new['phi_deg']:.2f}°")
    print(f"  중심 ({calib['cx']:.1f}, {calib['cy']:.1f}) → ({new['cx']:.1f}, {new['cy']:.1f})   "
          f"사진별 중심 흩어짐(표준편차) {ctrs.std(axis=0).round(1).tolist()} px")
    print(f"  카메라 고도각 약 {new['camera_elevation_deg']:.1f}°")


def main():
    ap = argparse.ArgumentParser(description="카메라 보정 (테두리 클릭 → STL 로 다듬기)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("click", help="테두리 클릭으로 타원 초기값")
    a1.add_argument("--camera", action="store_true", help="카메라로 바로 촬영해서 사용")
    a1.add_argument("--image", default=None, help="보정 사진 (--camera 면 저장 경로)")
    a1.add_argument("--points", default=None, help='"x1,y1 x2,y2 ..." (클릭 창 없이)')
    a2 = sub.add_parser("refine", help="STL 로 타원 다듬기")
    a2.add_argument("images", nargs="+", help="부품을 여러 각도로 돌려 찍은 사진")
    a2.add_argument("--shape", action="store_true", help="(실험) 타원 모양(a, b, 기울기)까지 다듬기")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    # kill / 터미널 닫기(SIGTERM, SIGHUP)로 끝나도 hik.grab 의 finally 가 돌아 카메라를 놓아 주게 한다
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda n, f: sys.exit(128 + n))
    try:
        if args.cmd == "click":
            cmd_click(args, cfg)
        else:
            cmd_refine(args, cfg)
    finally:
        cv2.destroyAllWindows()                              # 클릭 창이 남지 않게
        if "hik" in sys.modules:                             # 혹시 grab 이 중간에 끊겨도 SDK 를 확실히 정리
            try:
                sys.modules["hik"].sdk().MvCamera.MV_CC_Finalize()
            except Exception:
                pass


if __name__ == "__main__":
    main()
