#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정확도 검증: 부품을 '알고 있는 각도'만큼 돌려 가며 찍고, 측정값이 그만큼 바뀌는지 확인한다.
(회전 스테이지가 있으면 스테이지로, 없으면 부품 밑에 인쇄한 각도기를 깔고 위에서 볼 때 시계방향으로 돌린다)

1) 촬영
     python3 validate.py capture --out images/val --step 30 [--repeat 2]
   - 0°, 30°, 60° ... 순서로 안내한다. 부품을 그 각도에 맞추고 Enter → images/val/rot_030.png 저장 + 바로 측정 결과
   - --repeat N : 같은 자세에서 N장 (rot_030_r1.png ...) → 반복성 확인
   - --step 0   : 돌리지 않고 한 자세만 (--repeat 와 같이 써서 반복성만)
2) 분석
     python3 validate.py analyze images/val [--debug-dir images/val/debug]
   - 파일 이름의 각도(rot_<각도>.png / rot_<각도>_r<번호>.png)를 정답으로 써서 자세별 오차 표를 낸다
   - 각도기의 0° 와 측정의 0° 는 서로 모르므로 '공통 오프셋'을 빼고 비교한다 (= 상대 회전량의 정확도)
   - 문양 쪽이 180° 뒤집혀 판정된 장수도 센다
   - 결과는 <폴더>/results.csv 에도 저장
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))                              # side_plate_notch, stl_model
sys.path.insert(0, str(HERE))

import common as C                                          # noqa: E402
import side_plate_notch as SP                               # noqa: E402

NAME_RE = re.compile(r"^rot_(-?\d+(?:\.\d+)?)(?:_r(\d+))?\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


def cmd_capture(args, cfg):
    import hik
    calib = C.load_calib(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    angles = [0.0] if args.step <= 0 else list(np.arange(0, 360 - 1e-6, args.step))
    ctx = None
    print("부품을 각 각도로 맞추고 Enter (q 입력 후 Enter = 그만)")
    for ang in angles:
        for rep in range(1, args.repeat + 1):
            name = f"rot_{int(round(ang)):03d}" + (f"_r{rep}" if args.repeat > 1 else "") + ".png"
            msg = f"[{ang:5.1f}°]" + (f" {rep}/{args.repeat}번째" if args.repeat > 1 else "") + " 준비되면 Enter: "
            if input(msg).strip().lower() == "q":
                print("그만합니다.")
                return
            try:
                img = hik.grab(cfg)
            except RuntimeError as e:
                sys.exit(f"카메라 오류: {e}")
            cv2.imwrite(str(out / name), img)
            if ctx is None:
                ctx = SP.Context(cfg, calib, img.shape)
            r = SP.measure(img, ctx, want_debug=True)
            dbg = out / "debug"
            dbg.mkdir(exist_ok=True)
            cv2.imwrite(str(dbg / name.replace(".png", ".jpg")), SP.draw_debug(img, ctx, r))
            r.pop("_debug", None)
            print(f"  저장 {out / name}")
            print(SP.summary_kr(r))
    print(f"다음: python3 validate.py analyze {out}")


def circmean(deg):
    t = np.radians(np.asarray(deg, float))
    return math.degrees(math.atan2(np.sin(t).mean(), np.cos(t).mean()))


def cmd_analyze(args, cfg):
    calib = C.load_calib(cfg)
    folder = Path(args.folder)
    files = sorted(p for p in folder.iterdir() if NAME_RE.match(p.name))
    if not files:
        sys.exit(f"{folder} 에 rot_<각도>.png 파일이 없습니다.")
    rows, ctx = [], None
    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
    for p in files:
        mt = NAME_RE.match(p.name)
        nominal, rep = float(mt.group(1)), int(mt.group(2) or 1)
        img = cv2.imread(str(p))
        if ctx is None:
            ctx = SP.Context(cfg, calib, img.shape)
        r = SP.measure(img, ctx, want_debug=bool(dbg))
        if dbg:
            cv2.imwrite(str(dbg / (p.stem + ".jpg")), SP.draw_debug(img, ctx, r))
        r.pop("_debug", None)
        row = dict(file=p.name, nominal=nominal, rep=rep, status=r["status"])
        if r["status"] != "not_found":
            row.update(pattern=r["current"]["pattern_side_center_deg"], notch=r["current"]["notch_deg"][0],
                       margin=r["decision"]["margin"], peak=r["match"]["peak_ratio"], score=r["match"]["score"])
        rows.append(row)
        print(f"  {p.name:18s} {SP.STATUS_KR.get(r['status'], r['status']):10s} "
              + ("" if "pattern" not in row else
                 f"문양 쪽 가운데 {row['pattern']:7.2f}°  빨간 홈 {row['notch']:7.2f}°  "
                 f"판정 차 {row['margin'] * 100:4.1f}%  봉우리비 {row['peak']:.2f}"))

    ok = [r for r in rows if "pattern" in r]
    if len(ok) < 2:
        sys.exit("측정된 사진이 2장 미만이라 분석할 수 없습니다.")
    # 돌린 방향(시계/반시계)과 공통 오프셋 추정: 문양 쪽 가운데 = 오프셋 + 방향 x 돌린 각도
    best = None
    for sgn in (1, -1):
        off = circmean([r["pattern"] - sgn * r["nominal"] for r in ok])
        err = [C.wrap180(r["pattern"] - sgn * r["nominal"] - off) for r in ok]
        cost = sum(min(abs(e), 180 - abs(e)) for e in err)
        if best is None or cost < best[0]:
            best = (cost, sgn, off)
    _, sgn, off = best
    flips = 0
    for r in ok:
        e = C.wrap180(r["pattern"] - sgn * r["nominal"] - off)
        r["flip"] = abs(e) > 90                             # 문양 쪽이 반대로 판정됨
        flips += r["flip"]
        r["err"] = C.wrap180(e - 180) if r["flip"] else e   # 각도 오차 (뒤집힘과 따로)
    # 뒤집힘을 뺀 뒤 오프셋 다시
    off2 = circmean([r["pattern"] - sgn * r["nominal"] - (180 if r["flip"] else 0) for r in ok])
    for r in ok:
        r["err"] = C.wrap180(r["pattern"] - sgn * r["nominal"] - (180 if r["flip"] else 0) - off2)
    errs = np.array([r["err"] for r in ok])
    print()
    print(f"돌린 방향: {'시계방향' if sgn > 0 else '반시계방향'} (자동 판별),  공통 오프셋 {off2:.2f}°")
    print(f"각도 오차 (공통 오프셋 뺀 뒤): 평균 절대 {np.abs(errs).mean():.2f}°,  RMS {math.sqrt((errs ** 2).mean()):.2f}°,"
          f"  최대 {np.abs(errs).max():.2f}°   ({len(ok)}장)")
    print(f"문양 쪽 180° 오판: {flips}/{len(ok)}장")
    print(f"측정 실패: {len(rows) - len(ok)}/{len(rows)}장")
    # 반복성
    groups = {}
    for r in ok:
        groups.setdefault(r["nominal"], []).append(r["err"])
    sds = [np.std(v) for v in groups.values() if len(v) > 1]
    if sds:
        print(f"반복성 (같은 자세 여러 장의 표준편차): 평균 {np.mean(sds):.3f}°, 최대 {np.max(sds):.3f}°")
    print("자세별:")
    for nom in sorted(groups):
        v = groups[nom]
        print(f"  {nom:6.1f}°  오차 " + ", ".join(f"{e:+.2f}°" for e in v))
    with open(folder / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["파일", "돌린각도", "반복", "상태", "문양쪽가운데", "빨간홈", "판정차", "봉우리비", "각도오차", "문양뒤집힘"])
        for r in rows:
            w.writerow([r["file"], r["nominal"], r["rep"], r["status"], r.get("pattern", ""), r.get("notch", ""),
                        r.get("margin", ""), r.get("peak", ""), round(r["err"], 3) if "err" in r else "",
                        r.get("flip", "")])
    print(f"저장: {folder / 'results.csv'}")


def main():
    ap = argparse.ArgumentParser(description="정확도 검증 (촬영 / 분석)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("capture", help="안내에 따라 돌려 가며 촬영")
    a1.add_argument("--out", default=str(HERE / "images" / "val"))
    a1.add_argument("--step", type=float, default=30.0)
    a1.add_argument("--repeat", type=int, default=1)
    a2 = sub.add_parser("analyze", help="찍은 사진 분석")
    a2.add_argument("folder")
    a2.add_argument("--debug-dir", default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    if args.cmd == "capture":
        cmd_capture(args, cfg)
    else:
        cmd_analyze(args, cfg)


if __name__ == "__main__":
    main()
