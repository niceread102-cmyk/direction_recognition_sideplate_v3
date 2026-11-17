#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoneo 2D 보정 (카메라/ROI/키우기 배율을 바꿨을 때 1회). 보정 방법은 ../calibrate.py 와 같다.
결과: photoneo/calib.yaml (camera: 기록에 지금 Photoneo 설정이 남는다)

1단계: 원판 테두리 → 타원
  python3 photoneo/calibrate_pho.py click --auto     # 자동: 부품 테두리 원을 찾는다 (안쪽 금속이 바깥보다 어두운 원)
  python3 photoneo/calibrate_pho.py click            # 수동: 원판 윗면 바깥 테두리 8~12점 클릭 → Enter
  (서버가 켜져 있으면 서버로 찍고, 꺼져 있으면 카메라에 직접 접속)
  - 자동은 카메라가 거의 바로 위에서 내려다볼 때(지금 설치) 쓴다. 타원이 아니라 원으로 잡는다.

2단계: STL 로 다듬기 (선택, 부품을 여러 각도로 돌려 찍은 사진)
  python3 photoneo/client.py grab photoneo/images/rot_000.png    # 돌려 가며 여러 장
  python3 photoneo/calibrate_pho.py refine photoneo/images/rot_*.png

보정 후 서버가 켜져 있으면 자동으로 다시 읽게 한다 (client.py reload).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "hik"))                      # common/decision/matcher/calibrate
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import calibrate as CAL                                     # noqa: E402
import client                                               # noqa: E402
import common as C                                          # noqa: E402
import matcher as M                                         # noqa: E402
import pho                                                  # noqa: E402


def capture(cfg, path):
    """서버가 있으면 서버로, 없으면 직접 찍기 → (영상, 카메라 기록)"""
    try:
        r = client.request({"cmd": "grab", "save": str(path)}, timeout=30)
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        print(f"서버로 촬영: {path}")
        return cv2.imread(str(path)), r["camera"]
    except RuntimeError as e:
        if "연결하지 못했습니다" not in str(e):
            raise
    with pho.PhoxiCamera(cfg) as cam:
        img = cam.grab()
        cv2.imwrite(str(path), img)
        print(f"직접 촬영: {path}")
        return img, cam.tag


def find_rim(gray):
    """
    부품 윗면 바깥 테두리 원 찾기 → (cx, cy, r, 테두리 세기, 안팎 밝기 차)
    점수 = 원 위에서 '바깥으로 밝아지는' 경계 세기(반지름 방향 기울기 중앙값) / 접선 방향 기울기 중앙값
      - 각도기 눈금(반지름 방향 선)은 접선 방향 기울기라 점수가 낮다
      - 인쇄된 원은 안쪽이 밝아서 '안쪽이 바깥보다 어두움(금속 < 종이)' 조건에서 빠진다
      - 한쪽 그림자는 중앙값이라 영향이 작다
    Hough 원 여러 개를 출발점으로 중심 ±8%, 반지름 0.8~1.2 배를 훑고 마지막에 0.25 px 단위로 다듬는다 (1/4 해상도).
    """
    H, W = gray.shape
    d = 4
    g = cv2.GaussianBlur(cv2.resize(gray, None, fx=1 / d, fy=1 / d, interpolation=cv2.INTER_AREA).astype(np.float32),
                         (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1) / 8
    h, w = g.shape
    s = 800.0 / max(H, W)
    sm = cv2.GaussianBlur(cv2.resize(gray, None, fx=s, fy=s), (0, 0), 2)
    circ = cv2.HoughCircles(sm, cv2.HOUGH_GRADIENT, 1.5, 40, param1=60, param2=25,
                            minRadius=int(0.06 * 800), maxRadius=int(0.35 * 800))
    if circ is None:
        raise SystemExit("원판을 찾지 못했습니다. --auto 없이 클릭으로 보정하세요.")
    a = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)

    def samp(M_, x, y):
        return M_[np.clip(np.round(y).astype(int), 0, h - 1), np.clip(np.round(x).astype(int), 0, w - 1)]

    def score(cx, cy, r):
        x, y = cx + r * ca, cy + r * sa
        sx, sy = samp(gx, x, y), samp(gy, x, y)
        edge = np.median(sx * ca + sy * sa) / (np.median(np.abs(-sx * sa + sy * ca)) + 0.3)
        contrast = np.median(samp(g, cx + 1.06 * r * ca, cy + 1.06 * r * sa)) - \
            np.median(samp(g, cx + 0.88 * r * ca, cy + 0.88 * r * sa))
        return edge, contrast

    best = None
    for hx, hy, hr in circ[0][:10] / s / d:
        for f in np.arange(0.80, 1.21, 0.02):
            r = hr * f
            for dx in np.arange(-0.08, 0.081, 0.02) * r:
                for dy in np.arange(-0.08, 0.081, 0.02) * r:
                    e, c = score(hx + dx, hy + dy, r)
                    if c > 8 and (best is None or e > best[0]):
                        best = (e, c, hx + dx, hy + dy, r)
    if best is None:
        raise SystemExit("안쪽이 어두운 원(부품)을 찾지 못했습니다. --auto 없이 클릭으로 보정하세요.")
    e, c, cx, cy, r = best
    for step in (0.5, 0.25):
        improved = True
        while improved:
            improved = False
            for dx, dy, dr in ((step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                e2, c2 = score(cx + dx, cy + dy, r + dr)
                if c2 > 8 and e2 > e:
                    e, c, cx, cy, r, improved = e2, c2, cx + dx, cy + dy, r + dr, True
    return (cx + 0.5) * d - 0.5, (cy + 0.5) * d - 0.5, r * d, float(e), float(c)


def auto_calib(img, cfg):
    """
    자동 보정: 부품 테두리 원(find_rim)으로 중심과 크기를 정하고, 그 크기로 STL 맞추기 점수를 확인용으로 남긴다.
    (STL 점수만으로 크기까지 찾으면 점수가 크기에 둔감해 엉뚱한 원을 고를 때가 있었다)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    cx, cy, R, edge, contrast = find_rim(gray)
    print(f"  테두리 원: 중심 ({cx:.1f}, {cy:.1f}) 반지름 {R:.1f}px  테두리 세기 {edge:.1f}  안팎 밝기 차 {contrast:.0f}")
    if edge < 5:
        print("경고: 테두리가 뚜렷하지 않습니다. 확인 그림을 보고, 이상하면 클릭으로 보정하세요.")
    calib = dict(cx=float(cx), cy=float(cy), a=float(R), b=float(R), phi_deg=0.0, image_size=[W, H])
    model = C.load_model(cfg)
    cam = M.Camera(calib, model, gray.shape)
    vt = M.VisibilityTable(model, cam, 1.0, C.vis_cache_path(cfg, calib))
    pose = M.find_pose(gray, model, cam, vt, dict(cfg.get("matching", {}), scale_search=0.0))
    print(f"  STL 확인: 점수 {pose['score']:.3f}, 봉우리비 {pose['peak_ratio']:.2f}, "
          f"STL 중심과 테두리 중심 차 {np.hypot(*(np.asarray(pose['center']) - [cx, cy])):.1f}px")
    return dict(calib, camera_elevation_deg=90.0, rim_edge=round(edge, 2), rim_contrast=round(contrast, 1),
                stl_score=round(float(pose["score"]), 4), stl_peak_ratio=round(float(pose["peak_ratio"]), 3))


def tell_server_reload():
    try:
        r = client.request({"cmd": "reload"}, timeout=120)
        print("서버가 새 보정을 읽었습니다." if r.get("ok") else f"서버 다시 읽기 실패: {r.get('error')}")
    except RuntimeError:
        pass                                                 # 서버가 꺼져 있으면 다음에 켤 때 읽는다


def cmd_click(args, cfg):
    path = Path(args.image or (HERE / "images" / "calib_click.png")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    img, tag = capture(cfg, path)
    if args.auto:
        calib = auto_calib(img, cfg)
        calib["source"] = "테두리 자동 (원)"
    else:
        pts = CAL.click_points(img)
        h, w = img.shape[:2]
        calib = CAL.fit_ellipse(pts, (w, h))
        calib["source"] = "테두리 클릭"
    calib["camera"] = tag
    try:                                                     # 각도기 기준 각도 보정은 테두리 보정과 따로라 이어받는다
        old = C.load_calib(cfg)
        for k_ in ("frame_offset_deg", "frame_source", "notch_offset_deg", "notch_readings"):
            if k_ in old:
                calib[k_] = old[k_]
    except SystemExit:
        pass
    vis = img.copy()
    cv2.ellipse(vis, ((calib["cx"], calib["cy"]), (2 * calib["a"], 2 * calib["b"]), calib["phi_deg"]),
                (0, 255, 0), 3)
    cv2.drawMarker(vis, (int(calib["cx"]), int(calib["cy"])), (0, 0, 255), cv2.MARKER_CROSS, 60, 3)
    vp = path.with_name(path.stem + "_fit.jpg")
    cv2.imwrite(str(vp), vis)
    out = CAL.save_calib(cfg, calib)
    print(f"저장: {out}  (확인 그림: {vp})")
    print(f"  중심 ({calib['cx']:.1f}, {calib['cy']:.1f})  반축 a={calib['a']:.1f}px b={calib['b']:.1f}px  "
          f"기울기 {calib['phi_deg']:.1f}°  카메라 고도각 약 {calib['camera_elevation_deg']:.1f}°")
    tell_server_reload()


def cmd_refine(args, cfg):
    CAL.cmd_refine(args, cfg)                               # 기존 보정 파일의 camera 기록은 그대로 남는다
    tell_server_reload()


def pose_of(img, cfg, calib):
    """서버와 같은 방법으로 자세 찾기 → (ctx, pose)"""
    import server
    import side_plate_notch as SPN
    ctx = SPN.Context(cfg, calib, img.shape)
    ctx.cam.calib_c0 = ctx.cam.c0.copy()
    c0, _, _ = server.locate_part(img, ctx)
    ctx.cam.c0 = np.asarray(c0, float)
    pose = M.find_pose(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ctx.model, ctx.cam, ctx.vt, cfg.get("matching", {}))
    return ctx, pose


def cmd_pattern_ref(args, cfg):
    """
    문양 쪽 기준 띠 영상 만들기 (photoneo/pattern_ref.npz).
    첫 사진: STL 문양 위치를 초록 점으로 그린 확인 그림을 보고, 초록 격자가 실제 각인 격자 위에 있는지 답한다.
    나머지 사진: 지금까지의 기준과 비교해 방향을 자동으로 맞춰 평균한다 (부품을 여러 각도로 놓고 찍으면 더 튼튼).
    """
    import pattern_ref as PR
    calib = C.load_calib(cfg)
    imgs = []
    if args.image:
        for p in args.image:
            im = cv2.imread(p)
            if im is None:
                sys.exit(f"사진을 읽을 수 없습니다: {p}")
            imgs.append((p, im))
    else:
        out = HERE / "images" / "pattern_ref_1.png"
        imgs.append((str(out), capture(cfg, out)[0]))
    samples = []
    for i, (p, im) in enumerate(imgs):
        ctx, pose = pose_of(im, cfg, calib)
        P = PR.unwrap(im, ctx, pose["psi"], pose["center"], pose["k"])
        ok = None
        if i == 0:
            vis = im.copy()
            u, v = ctx.cam.project(ctx.model.p[ctx.model.is_pattern], pose["psi"], pose["center"], pose["k"])
            for x, y in zip(u.astype(int), v.astype(int)):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
            vp = HERE / "images" / "pattern_ref_check.jpg"
            cv2.imwrite(str(vp), vis)
            if args.first_ok is not None:
                ok = args.first_ok == "y"
            else:
                print(f"확인 그림: {vp}")
                ans = input("초록 점(STL 문양)이 사진의 각인 격자/숫자 위에 있습니까? [y/n] ").strip().lower()
                ok = ans.startswith("y")
        samples.append((P, ok))
        print(f"  {Path(p).name}: 모델 회전 {pose['psi']:.1f}°  STL 점수 {pose['score']:.3f}")
    ref, log = PR.build(samples)
    for i, flipped, sim in log:
        print(f"  {Path(imgs[i][0]).name}: {'180° 뒤집어' if flipped else '그대로'} 사용"
              + ("" if sim is None else f" (기준과 닮음 {sim:.3f})"))
    out = HERE / "pattern_ref.npz"
    if out.exists():
        out.replace(out.with_suffix(".npz.bak"))
    PR.save(out, ref, len(samples), note=", ".join(Path(p).name for p, _ in imgs))
    print(f"저장: {out}  (사진 {len(samples)}장)")
    tell_server_reload()


def cmd_frame(args, cfg):
    """
    각도기 기준 각도 보정: 인쇄한 각도기의 큰 점 12개로 각도기 0° 가 화면 위쪽에서 몇 도 돌아가 있는지 재서
    보정 파일에 frame_offset_deg(= −회전)로 저장한다. 이후 모든 각도 출력이 각도기(=기준) 각도가 된다.
    각도기 종이(또는 스테이지)를 카메라에 대해 움직이면 다시 한다.
    """
    import protractor as PT
    path = Path(args.image or (HERE / "images" / "frame.png")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(path)) if args.image else capture(cfg, path)[0]
    if img is None:
        sys.exit(f"사진을 읽을 수 없습니다: {path}")
    prot = PT.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if prot is None:
        sys.exit("각도기의 큰 점을 8개 이상 찾지 못했습니다. 각도기가 화면 안에 다 보이는지 확인하세요.")
    vis = img.copy()
    for x, y in prot["dots"]:
        cv2.circle(vis, (int(x), int(y)), 30, (0, 0, 255), 4)
    c = np.array(prot["center"])
    M_ = np.array(prot["M"])
    for k in range(12):                                      # 맞춘 각도기의 0/30/60.. 방향 선
        t = np.radians(30 * k)
        e = c + M_ @ np.array([np.sin(t), -np.cos(t)])
        cv2.line(vis, tuple(c.astype(int)), tuple(e.astype(int)), (0, 255, 0) if k else (255, 0, 255), 3)
    vp = path.with_name(path.stem + "_check.jpg")
    cv2.imwrite(str(vp), vis)
    calib = C.load_calib(cfg)
    calib.update(frame_offset_deg=round(-prot["rotation_deg"], 4),
                 frame_source=f"각도기 점 {prot['n_dots']}개 (잔차 {prot['rms_px']:.1f}px, 축비 {prot['axis_ratio']:.4f})")
    out = CAL.save_calib(cfg, calib)
    print(f"각도기 회전 {prot['rotation_deg']:+.3f}° (화면 위쪽 기준 시계방향 +)  → 보정 frame_offset_deg {-prot['rotation_deg']:+.3f}°")
    print(f"  점 {prot['n_dots']}개, 잔차 {prot['rms_px']:.1f}px, 인쇄 축비 {prot['axis_ratio']:.4f} "
          f"(1 에서 먼 만큼 각도기 눈금 자체가 방향에 따라 최대 약 {(1 - prot['axis_ratio']) / 2 * 57.3:.1f}° 틀림)")
    print(f"저장: {out}  (확인 그림: {vp}, 보라색 선 = 각도기 0°)")
    tell_server_reload()


def cmd_notch(args, cfg):
    """
    빨간 홈 위치 보정 (각도기 판독값으로).
    빨간 홈은 카메라에 보이지 않아 STL 설계 위치(긴 홀 기준)를 쓴다. 실제 부품의 홈이 설계와 몇 도 다르면
    각도가 그만큼 일정하게 틀린다. 부품을 각도기 위에 놓고 A, B 홈을 각도기 눈금으로 읽어 넣으면
    서버가 계산한 '각도기로 읽으면' 값(부품 중심 어긋남, 인쇄 찌그러짐 포함)과 비교해 보정값을 구한다.
    여러 자세를 넣을수록 판독 오차가 평균으로 줄어든다.
    """
    calib = C.load_calib(cfg)
    entries = list(calib.get("notch_readings", []))
    if args.reset:
        entries = []
        print("지금까지 넣은 판독값을 지웠습니다.")
    if args.a is not None or args.b is not None:
        if args.a is None or args.b is None:
            sys.exit("--a 와 --b 를 함께 넣으세요 (지금 자세에서 각도기로 읽은 빨간 홈 A, B).")
        r = client.request({"cmd": "measure"}, timeout=60)
        if not r.get("ok"):
            sys.exit(f"측정 실패: {r.get('error')}")
        pc = r["result"].get("protractor")
        if not pc:
            sys.exit("화면에서 각도기를 찾지 못했습니다. 각도기가 다 보이게 놓으세요.")
        pa, pb = pc["notch_reading_deg"]["A"], pc["notch_reading_deg"]["B"]
        da, db = C.wrap180(args.a - pa), C.wrap180(args.b - pb)
        entries.append(dict(read_a=float(args.a), read_b=float(args.b), pred_a=pa, pred_b=pb,
                            d_a=round(da, 3), d_b=round(db, 3), time=time_str()))
        print(f"이번 자세: 서버 계산 A {pa:.1f}° / B {pb:.1f}°, 읽은 값 A {args.a:.1f}° / B {args.b:.1f}°"
              f"  → 차 A {da:+.2f}°, B {db:+.2f}°")
        if abs(da - db) > 2.0:
            print("  주의: A 와 B 의 차가 2° 넘게 다릅니다. 판독을 다시 확인하세요 (A/B 를 바꿔 읽었는지 등).")
    old = float(calib.get("notch_offset_deg", 0.0))
    if entries:
        d = np.array([(e["d_a"] + e["d_b"]) / 2 for e in entries])
        # 판독 차 d(시계 +) 만큼 실제 홈이 더 시계방향 → 모델 각도(반시계 +)는 그만큼 빼고, 이미 넣은 보정에 더한다
        new = old - float(d.mean())
        print(f"판독 {len(entries)}자세: 평균 차 {d.mean():+.2f}°, 흩어짐(표준편차) {d.std():.2f}°, 최대 {np.abs(d).max():.2f}°")
        for i, e in enumerate(entries, 1):
            print(f"  {i}. A 읽음 {e['read_a']:.1f} / 계산 {e['pred_a']:.1f} ({e['d_a']:+.2f})   "
                  f"B 읽음 {e['read_b']:.1f} / 계산 {e['pred_b']:.1f} ({e['d_b']:+.2f})")
        if args.apply:
            calib.update(notch_offset_deg=round(new, 3), notch_readings=[])
            print(f"홈 위치 보정 {old:+.3f}° → {new:+.3f}° 적용 (판독값은 비움: 이제부터 넣는 값은 새 보정 기준)")
        else:
            calib["notch_readings"] = entries
            print(f"적용하면 홈 위치 보정 {old:+.3f}° → {new:+.3f}°.  적용: python3 photoneo/calibrate_pho.py notch --apply")
    else:
        print(f"넣은 판독값이 없습니다. 지금 홈 위치 보정 {old:+.3f}°")
    CAL.save_calib(cfg, calib)
    tell_server_reload()


def time_str():
    import time as _t
    return _t.strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description="Photoneo 2D 보정")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("click", help="테두리 → 타원 (--auto: STL 로 자동)")
    a1.add_argument("--auto", action="store_true", help="STL 맞추기로 자동")
    a1.add_argument("--image", default=None, help="촬영 사진 저장 경로")
    a2 = sub.add_parser("refine", help="STL 로 다듬기")
    a2.add_argument("images", nargs="+")
    a2.add_argument("--shape", action="store_true", help="(실험) 타원 모양까지 다듬기")
    a3 = sub.add_parser("pattern-ref", help="문양 쪽 기준 띠 영상 만들기")
    a3.add_argument("--image", nargs="+", default=None, help="사진 파일들 (없으면 지금 1장 촬영)")
    a3.add_argument("--first-ok", choices=["y", "n"], default=None,
                    help="첫 사진에서 STL 문양 위치가 맞는지 (묻지 않고)")
    a4 = sub.add_parser("frame", help="각도기 기준 각도 보정 (각도기 회전 → frame_offset_deg)")
    a4.add_argument("--image", default=None, help="사진 파일 (없으면 지금 촬영)")
    a5 = sub.add_parser("notch", help="빨간 홈 위치 보정 (각도기로 읽은 A, B 홈 각도로)")
    a5.add_argument("--a", type=float, default=None, help="지금 자세에서 각도기로 읽은 빨간 홈 A 각도")
    a5.add_argument("--b", type=float, default=None, help="지금 자세에서 각도기로 읽은 빨간 홈 B 각도")
    a5.add_argument("--apply", action="store_true", help="지금까지 넣은 판독값의 평균으로 보정 적용")
    a5.add_argument("--reset", action="store_true", help="넣은 판독값 지우기")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    try:
        {"click": cmd_click, "refine": cmd_refine, "pattern-ref": cmd_pattern_ref,
         "frame": cmd_frame, "notch": cmd_notch}[args.cmd](args, cfg)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
