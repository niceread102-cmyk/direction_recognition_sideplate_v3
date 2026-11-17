#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoneo 2D 측정 상주 서버

한 번 켜 두면 Photoneo 접속(약 5 s)과 STL/카메라 모델 준비를 다시 하지 않는다.
요청이 오면 소프트웨어 트리거로 찍고(약 0.2 s) 바로 측정해서 돌려준다.
측정 알고리즘은 Hikvision 2D 카메라와 같다 (../side_plate_notch.py 의 measure).

켜기
  python3 photoneo/server.py                 # 기본 127.0.0.1:5577 (photoneo/config.yaml 의 server)
  python3 photoneo/server.py --port 5600

요청 (client.py 를 쓰면 된다)
  TCP 로 JSON 한 줄을 보내면 JSON 한 줄이 돌아온다.
    {"cmd": "measure", "debug": "/절대/경로.jpg", "save": "/절대/경로.png"}   → 찍고 측정
    {"cmd": "grab", "save": "/절대/경로.png", "raw": false}                  → 찍어서 저장만
    {"cmd": "reload"}   → config.yaml / calib.yaml 다시 읽기 (보정을 새로 한 뒤)
    {"cmd": "ping"}     {"cmd": "stop"}
  응답: {"ok": true, "result": {...측정 결과...}, "summary": "한글 요약", "time": {"grab_s":..,"measure_s":..}}
        {"ok": false, "error": "..."}
"""
from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "hik"))                      # common/decision/matcher/calibrate
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import common as C                                          # noqa: E402
import matcher as M                                         # noqa: E402
import numpy as np                                          # noqa: E402
import pattern_ref as PR                                    # noqa: E402
import pho                                                  # noqa: E402
import protractor as PT                                     # noqa: E402
import side_plate_notch as SPN                              # noqa: E402


def log(msg):
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def locate_part(img, ctx, last=None):
    """
    부품 중심 찾기 (부품을 보정 때와 다른 자리에 놓아도 되게).
    측정의 STL 맞추기는 보정 중심 ±center_search_px 안만 찾는다. 그래서 먼저
      후보 = 보정 중심, 지난번 중심, 보정 크기와 비슷한 Hough 원 몇 개
    마다 거친 STL 점수(1/4 해상도)를 매겨 가장 좋은 곳을 중심 초기값으로 쓴다.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cam = ctx.cam
    R = float(np.sqrt(abs(np.linalg.det(cam.A))))           # 원판 반지름 (px)
    cands = [np.array(cam.calib_c0, float)]
    if last is not None:
        cands.append(np.array(last, float))
    s = 0.125
    sm = cv2.GaussianBlur(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    circ = cv2.HoughCircles(sm, cv2.HOUGH_GRADIENT, 1.2, R * s * 0.5, param1=60, param2=20,
                            minRadius=int(R * s * 0.85), maxRadius=int(R * s * 1.15))
    if circ is not None:
        cands += [np.array([x, y]) / s for x, y, _ in circ[0][:4]]
    uniq = []
    for c in cands:                                          # 30 px 안에 겹치는 후보는 하나만
        if all(np.hypot(*(c - u)) >= 30 for u in uniq):
            uniq.append(c)
    grad = M.Gradient(gray, sigma=3.0, down=4)
    # 원판 테두리 확인: 반지름 R 근처 원 위에서 밝기 경계가 '원 방향'(반지름 방향 기울기)으로 나 있는 정도.
    # 실제 부품 테두리는 7~11, 각도기 눈금(반지름 방향 선)이나 빈 종이는 1 근처라 STL 점수보다 확실히 갈린다.
    strength = [rim_strength(grad, c, R) for c in uniq]
    top = max(strength)
    if top >= 3.0:
        keep = [c for c, s_ in zip(uniq, strength) if s_ >= max(3.0, 0.5 * top)]
    else:
        keep = uniq                                          # 테두리가 뚜렷한 후보가 없으면 (배경이 다를 때) 모두
    g4 = type("G", (), {"g4": grad})()                       # coarse_search 는 1/4 영상만 쓴다
    # 후보 고르기만 하므로 성기게 (중심 ±24 px / 12 px 간격, 각도 6° 간격). 정밀한 것은 measure 의 find_pose 가 한다
    scored = [M.coarse_search(ctx.model, cam, ctx.vt, g4, c, 24.0, 6.0, 12.0, top_n=1)[0] for c in keep]
    sc, _, ctr = max(scored, key=lambda r: r[0])
    # 고른 후보 주변을 한 번 더 촘촘히 (중심 ±18 px / 6 px, 각도 3°) → find_pose 가 엉뚱한 곳에 빠지지 않게
    sc, _, ctr = M.coarse_search(ctx.model, cam, ctx.vt, g4, ctr, 18.0, 3.0, 6.0, top_n=1)[0]
    return ctr, top, len(keep)


def rim_strength(grad, c, R, n=180):
    """중심 c, 반지름 0.9R~1.1R 의 원 위에서 (반지름 방향 기울기 중앙값 / 접선 방향 기울기 중앙값) 의 최대"""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = np.cos(a), np.sin(a)
    best = 0.0
    for r in np.arange(0.9 * R, 1.1 * R, 4.0):
        gx, gy = grad.sample(c[0] + r * ca, c[1] + r * sa)
        rad = np.median(np.abs(gx * ca + gy * sa))
        tan = np.median(np.abs(-gx * sa + gy * ca))
        best = max(best, float(rad / (tan + 0.05)))
    return best


class Service:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = C.load_config(config_path)
        self.cam = pho.PhoxiCamera(self.cfg)
        self.ctx = None
        self.ctx_error = None
        self.pref = None
        self.frame_offset = 0.0
        self.reload(reopen_camera=False)

    def reload(self, reopen_camera=True):
        """설정/보정 다시 읽기. 카메라 설정이 바뀌었으면 카메라도 다시 설정"""
        new = C.load_config(self.config_path)
        cam_changed = new.get("camera") != self.cfg.get("camera")
        self.cfg = new
        self.cam.cfg = new
        if reopen_camera and cam_changed:
            log("카메라 설정이 바뀌어 다시 연결합니다.")
            self.cam.close()
            self.cam.open()
        self.ctx, self.ctx_error = None, None
        try:
            pho.check_calib(self.cfg, self.cam.tag)
            calib = C.load_calib(self.cfg)
            # 각도기 기준 각도 보정 (calibrate_pho.py frame): 모든 각도 출력에 더한다
            self.frame_offset = float(calib.get("frame_offset_deg", 0.0))
            self.cfg.setdefault("angles", {})["frame_offset_deg"] = self.frame_offset
            if self.frame_offset:
                log(f"각도기 기준 각도 보정 {self.frame_offset:+.3f}° ({calib.get('frame_source', '')})")
            img = self.cam.grab()                                # 영상 크기를 알아야 카메라 모델을 만든다
            t = time.time()
            self.ctx = SPN.Context(self.cfg, calib, img.shape)
            self.ctx.cam.calib_c0 = self.ctx.cam.c0.copy()
            self.last_center = None
            # 빨간 홈 위치 보정 (calibrate_pho.py notch): 실제 홈이 STL 설계 위치와 다른 만큼 (모델 각도, 반시계 +)
            self.notch_offset = float(calib.get("notch_offset_deg", 0.0))
            if self.notch_offset:
                self.ctx.notch = (self.ctx.notch + self.notch_offset) % 360
                pa = self.ctx.model.pattern_angle
                self.ctx.stl_pattern_half = None if pa is None else ("P" if (pa - self.ctx.notch) % 360 < 180 else "Q")
                log(f"빨간 홈 위치 보정 {self.notch_offset:+.3f}° (STL {self.ctx.model.notches[0]['angle']:.1f}° → {self.ctx.notch:.2f}°)")
            log(f"측정 준비 완료 (STL/보이는 점 표 {time.time() - t:.1f} s)  영상 {img.shape[1]}x{img.shape[0]}")
            self.pref = None
            d = self.cfg.get("decision", {})
            if d.get("method", "halves") == "reference":
                p = C.rel_path(self.cfg, d.get("reference_file", "pattern_ref.npz"))
                try:
                    self.pref = PR.PatternRef(p, d.get("reference_min_margin", 0.08))
                    log(f"문양 판정: 기준 사진 비교 ({p.name}, 사진 {self.pref.n}장)")
                except (OSError, RuntimeError, KeyError) as e:
                    log(f"문양 판정: 기준 사진을 쓸 수 없어({e}) 반쪽 비교로 하고, 결과는 모두 '문양 판정 불확실'로 냅니다. "
                        f"만들기: python3 photoneo/calibrate_pho.py pattern-ref")
        except (Exception, SystemExit) as e:                   # 보정 전이어도 grab 은 되게 서버는 켜 둔다
            self.ctx_error = str(e)
            log(f"측정 준비 안 됨: {e}")

    def grab(self, req):
        t = time.time()
        img = self.cam.grab(raw=bool(req.get("raw", False)))
        dt = time.time() - t
        out = {"ok": True, "shape": list(img.shape), "time": {"grab_s": round(dt, 3)}, "camera": self.cam.tag}
        if req.get("save"):
            p = Path(req["save"])
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img)
            out["saved"] = str(p)
        return out, img

    def measure(self, req):
        if self.ctx is None:
            return {"ok": False, "error": f"측정 준비가 안 됐습니다: {self.ctx_error}  "
                                          f"(보정: python3 photoneo/calibrate_pho.py click)"}
        out, img = self.grab(req)
        t = time.time()
        rim = None
        if self.cfg.get("matching", {}).get("locate_part", True):
            last = None if req.get("fresh") else self.last_center    # fresh: 지난번 중심을 쓰지 않음 (시험용)
            c0, rim, n = locate_part(img, self.ctx, last)
            self.ctx.cam.c0 = np.asarray(c0, float)          # find_pose 가 이 중심 ±center_search_px 를 찾는다
            out["time"]["locate_s"] = round(time.time() - t, 3)
            out["locate"] = {"center_px": [round(float(v), 1) for v in c0], "rim_strength": round(rim, 2),
                             "candidates": n}
        res = SPN.measure(img, self.ctx, want_debug=bool(req.get("debug")),
                          decide=self.pref.decide if self.pref else None)
        if self.cfg.get("decision", {}).get("method") == "reference" and self.pref is None and "rotate_cw_deg" in res:
            if res["status"] == "ok":                         # 기준 사진 없이 낸 문양 판정은 믿지 않는다
                res["status"] = "low_confidence"
            res["warning"] = ((res.get("warning", "") + " / ") if res.get("warning") else "") + \
                "기준 사진이 없어 반쪽 비교로 판정했습니다(믿기 어려움). python3 photoneo/calibrate_pho.py pattern-ref"
        if rim is not None and rim < 3.0 and res["status"] == "ok":
            res["status"] = "angle_uncertain"
            res["warning"] = ((res.get("warning", "") + " / ") if res.get("warning") else "") + \
                f"부품 테두리가 뚜렷하지 않아(테두리 세기 {rim:.1f}) 부품 위치를 잘못 잡았을 수 있습니다."
        if "match" in res and res["status"] != "not_found":
            self.last_center = res["match"]["center_px"]
        if "rotate_cw_deg" in res and self.cfg.get("angles", {}).get("protractor_check", True):
            res["protractor"] = self.protractor_check(img, res)
        out["time"]["measure_s"] = round(time.time() - t, 3)
        if req.get("debug"):
            p = Path(req["debug"])
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), SPN.draw_debug(img, self.ctx, res))
            out["debug"] = str(p)
        res.pop("_debug", None)
        summary = SPN.summary_kr(res)
        pc = res.get("protractor")
        if pc:
            summary += (f"\n  각도기로 읽으면: 빨간 홈 A {pc['notch_reading_deg']['A']:.1f}°, B {pc['notch_reading_deg']['B']:.1f}°"
                        f"  (부품 중심이 각도기 중심에서 {pc['part_offset_mm']:.1f} mm 벗어나 있어 위 측정값과 다르게 읽힘)"
                        f"\n  각도기: 회전 {pc['rotation_deg']:+.2f}°, 인쇄 축비 {pc['axis_ratio']:.3f}"
                        + (f"\n  경고: {pc['warning']}" if pc.get("warning") else ""))
        out.update(result=res, summary=summary)
        return out

    def protractor_check(self, img, res):
        """
        (검증용) 화면에 인쇄한 각도기가 보이면, 빨간 홈 A/B 가 그 각도기 눈금에서 몇 도로 읽혀야 하는지.
        부품이 각도기 중심에서 벗어나 있으면 눈금으로 읽는 값은 홈의 실제 방향(위 측정값)과 다르다.
        """
        prot = PT.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if prot is None:
            return None
        psi = res["match"]["model_rotation_deg"]
        ctr = res["match"]["center_px"]
        notch = self.ctx.notch
        img_deg = {k: (C.plane_from_model(notch + s, psi) + 90.0) % 360 for k, s in (("A", 0.0), ("B", 180.0))}
        c = np.array(prot["center"])
        out = dict(rotation_deg=round(prot["rotation_deg"], 3), axis_ratio=round(prot["axis_ratio"], 4),
                   n_dots=prot["n_dots"], center_px=[round(v, 1) for v in prot["center"]],
                   part_offset_mm=round(float(np.hypot(*(np.asarray(ctr) - c)) / prot["px_per_mm"]), 2),
                   notch_reading_deg={k: round(PT.reading(prot, ctr, a), 2) for k, a in img_deg.items()})
        diff = C.wrap180(-prot["rotation_deg"] - self.frame_offset)
        if abs(diff) > 0.3:
            out["warning"] = (f"각도기가 보정 때와 {diff:+.2f}° 다르게 놓여 있습니다. 각도기 기준으로 맞추려면 "
                              f"python3 photoneo/calibrate_pho.py frame")
        return out

    def handle(self, req):
        cmd = req.get("cmd")
        if cmd == "measure":
            return self.measure(req)
        if cmd == "grab":
            return self.grab(req)[0]
        if cmd == "reload":
            self.reload()
            return {"ok": self.ctx is not None, "error": self.ctx_error, "camera": self.cam.tag}
        if cmd == "ping":
            return {"ok": True, "camera": self.cam.tag, "ready": self.ctx is not None, "error": self.ctx_error}
        return {"ok": False, "error": f"모르는 명령: {cmd}"}

    def close(self):
        self.cam.close()


def serve(svc, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    log(f"대기 중 {host}:{port}  (끄기: Ctrl+C 또는 python3 photoneo/client.py stop)")
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                f = conn.makefile("rwb")
                line = f.readline()
                if not line:
                    continue
                req = json.loads(line.decode("utf-8"))
                if req.get("cmd") == "stop":
                    f.write(b'{"ok": true}\n')
                    f.flush()
                    log("종료 요청")
                    return
                t = time.time()
                try:
                    resp = svc.handle(req)
                except Exception as e:
                    traceback.print_exc()
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                resp.setdefault("time", {})["total_s"] = round(time.time() - t, 3)
                f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
                r = resp.get("result", {})
                log(f"{req.get('cmd')}: {'ok' if resp.get('ok') else '실패'}  {resp['time']}"
                    + (f"  돌릴 각도 {r['rotate_cw_deg']:+.2f}° ({r['status']})" if "rotate_cw_deg" in r else "")
                    + ("" if resp.get("ok") else f"  {resp.get('error')}"))
            except Exception as e:
                log(f"요청 처리 오류: {e}")


def main():
    ap = argparse.ArgumentParser(description="Photoneo 2D 측정 상주 서버")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    host = args.host or cfg.get("server", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("server", {}).get("port", 5577))
    # kill / 터미널 닫기로 끝나도 카메라를 놓아 주게 (finally 가 돌도록)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda n, f: sys.exit(128 + n))
    svc = Service(args.config)
    try:
        serve(svc, host, port)
    except KeyboardInterrupt:
        log("Ctrl+C")
    finally:
        svc.close()
        log("카메라 연결 종료")


if __name__ == "__main__":
    main()
