#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoneo PhoXi 3D Scanner 를 2D 카메라로 쓰기 (GigE Vision / harvesters + ImpactAcquire GenTL).
~/Projects/photoneo-python-examples/GigE-V/harvesters 예제와 같은 방법으로 접속한다.

2D 촬영
  CameraOnlyMode = True  : 패턴 투사(3D 스캔) 없이 카메라 영상만 찍는다 (한 장 약 0.2 s, 3D 스캔은 약 0.7 s)
  source: primary        : 3D 카메라(흑백 2472x2064, 12비트) + LED 조명   ← 기본, 가장 선명
  source: color          : 컬러 카메라 (camera.color_resolution)

빠르게 찍기
  PhoxiCamera 를 한 번 열어 두면 접속(약 4.5 s)과 설정을 다시 하지 않고, 소프트웨어 트리거 → 받기만 한다.
  server.py 가 이것을 계속 열어 두고 측정 요청을 받는다.

영상 처리 (측정 코드에 넘기기 전)
  rotate  : 화면 위쪽 = 0° 가 되도록 돌리기 (지금 설치는 카메라가 거꾸로 달려 180)
  roi     : 돌린 영상에서 [x, y, w, h] 만 잘라 쓰기 (부품 주변)
  upscale : 잘라낸 영상을 키우기. 측정 코드의 px 단위 설정(흐림 크기, 각인 선 필터 등)은
            원판 반지름 약 600 px 에 맞춰져 있어, 작게 찍히는 Photoneo 영상은 키워서 맞춘다.
  rotate/roi/upscale 을 바꾸면 다시 보정해야 한다 (보정 파일의 camera 기록과 다르면 멈춘다).

단독 사용
  python3 photoneo/pho.py list                       # 연결된 Photoneo 장치
  python3 photoneo/pho.py grab out.png               # 1장 찍어 저장 (설정대로 돌리기/자르기/키우기)
  python3 photoneo/pho.py grab out.png --raw         # 돌리기만 하고 자르기/키우기 없이 (ROI 정할 때)
  python3 photoneo/pho.py bench -n 10                # 연속 촬영 시간 재기
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def log(msg):
    print(f"[pho] {msg}", file=sys.stderr, flush=True)


def producer_path():
    """ImpactAcquire GenTL producer (예제의 default_gentl_producer.py 와 같은 방식)"""
    name = "mvGenTLProducer.cti" if sys.platform == "win32" else "libmvGenTLProducer.so"
    paths = [p for p in os.environ.get("GENICAM_GENTL64_PATH", "").split(os.pathsep) if p]
    paths += ["/opt/mvIMPACT_Acquire/lib/arm64", "/opt/mvIMPACT_Acquire/lib/x86_64", "/opt/ImpactAcquire/lib/arm64",
              "/opt/ImpactAcquire/lib/x86_64"]
    for p in paths:
        f = Path(p) / name
        if f.exists():
            return f
    raise RuntimeError(f"GenTL producer({name}) 를 찾지 못했습니다. GENICAM_GENTL64_PATH 를 확인하세요 "
                       f"(지금: {os.environ.get('GENICAM_GENTL64_PATH')})")


def cam_cfg(cfg):
    return cfg.get("camera", {})


def camera_id(cfg, serial):
    """보정 파일에 남길 식별자: 영상 모양을 바꾸는 설정(소스, 돌리기, 자르기, 키우기)을 모두 넣는다"""
    c = cam_cfg(cfg)
    roi = c.get("roi")
    src = str(c.get("source", "primary"))
    if src == "color":
        src += f"({c.get('color_resolution', 'Resolution_3864x2192')})"
    return (f"photoneo:{serial}:{src}:rot={int(c.get('rotate', 0))}"
            f":roi={'none' if not roi else ','.join(str(int(v)) for v in roi)}"
            f":up={float(c.get('upscale', 1.0)):g}")


def check_calib(cfg, cam_tag):
    """보정 파일이 지금 카메라 설정으로 만든 것인지 (다르면 RuntimeError, 기록이 없으면 경고)"""
    p = Path(cfg["calibration"]["file"])
    if not p.is_absolute():
        p = Path(cfg.get("_dir", ".")) / p
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        tag = (yaml.safe_load(f) or {}).get("camera")
    if tag is None:
        log(f"{p.name} 에 카메라 기록이 없습니다. photoneo/calibrate_pho.py click 으로 보정하세요.")
    elif tag != cam_tag:
        raise RuntimeError(f"{p} 는 다른 카메라 설정으로 보정한 파일입니다.\n  보정: {tag}\n  지금: {cam_tag}\n"
                           f"photoneo/calibrate_pho.py click 으로 다시 보정하세요.")


def process(img, cfg, raw=False):
    """센서 영상 → 측정용 BGR (돌리기 → 자르기 → 키우기)"""
    c = cam_cfg(cfg)
    rot = int(c.get("rotate", 0)) % 360
    if rot not in _ROT:
        raise ValueError(f"camera.rotate 는 0/90/180/270 : {rot}")
    if _ROT[rot] is not None:
        img = cv2.rotate(img, _ROT[rot])
    if raw:
        return img
    roi = c.get("roi")
    if roi:
        x, y, w, h = (int(v) for v in roi)
        H, W = img.shape[:2]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"camera.roi {roi} 가 (돌린) 영상 {W}x{H} 밖으로 나갑니다.")
        img = img[y:y + h, x:x + w]
    up = float(c.get("upscale", 1.0))
    if abs(up - 1.0) > 1e-6:
        img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(img)


class PhoxiCamera:
    """한 번 열어 두고 계속 찍는 Photoneo 2D 카메라"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.h = self.ia = self.nm = None
        self.serial = None
        self.open()

    # ------------------------------------------------------------ 열기 / 설정
    def open(self):
        from harvesters.core import Harvester
        c = cam_cfg(self.cfg)
        t0 = time.time()
        self.h = Harvester()
        self.h.add_file(str(producer_path()), check_existence=True, check_validity=True)
        self.h.update()
        devs = [d.property_dict for d in self.h.device_info_list if "Photoneo" in str(d.property_dict.get("vendor", ""))]
        want = str(c.get("serial", "auto"))
        pick = next((d for d in devs if want.lower() == "auto" or d.get("serial_number") == want), None)
        if pick is None:
            self.h.reset()
            found = ", ".join(f"{d.get('model')}({d.get('serial_number')})" for d in devs) or "없음"
            raise RuntimeError(f"Photoneo serial={want} 를 찾지 못했습니다. 찾은 장치: {found}. "
                               f"랜 연결/IP 대역, PhoXi Control 이 장치를 잡고 있지 않은지 확인하세요.")
        self.serial = pick["serial_number"]
        try:
            self.ia = self.h.create({"serial_number": self.serial})
        except Exception as e:
            self.h.reset()
            raise RuntimeError(f"{self.serial} 접속 실패: {e}. PhoXi Control 등 다른 프로그램이 쓰고 있으면 닫으세요.")
        self.nm = self.ia.remote_device.node_map
        try:
            self.ia.data_streams[0].node_map.mvResendActive.value = True      # 예제 README: 패킷 재전송 (Timeout 방지)
        except Exception:
            pass
        self.configure()
        self.ia.start()
        log(f"{pick.get('model')} ({self.serial}) 연결, 2D 설정 완료 ({time.time() - t0:.1f} s)  "
            f"{self.nm.Width.value}x{self.nm.Height.value} {self.nm.PixelFormat.value}")

    def _set(self, name, value, required=False):
        try:
            self.nm.get_node(name).value = value
            return True
        except Exception as e:
            if required:
                raise RuntimeError(f"{name}={value} 설정 실패: {e}")
            log(f"{name}={value} 설정 실패 (무시): {e}")
            return False

    def configure(self):
        c = cam_cfg(self.cfg)
        nm = self.nm
        if c.get("load_default_user_set", True):
            nm.UserSetSelector.value = "Default"
            nm.UserSetLoad.execute()
        nm.TriggerSelector.value = "FrameStart"
        nm.TriggerMode.value = "On"
        nm.TriggerSource.value = "Software"
        self._set("CameraOnlyMode", bool(c.get("camera_only", True)), required=True)
        src = str(c.get("source", "primary"))
        if src == "primary":
            self._set("TextureSource", str(c.get("texture_source", "LED")), required=True)
            comp = "Intensity"
        elif src == "color":
            self._set("TextureSource", "Color", required=True)
            if c.get("color_resolution"):
                self._set("ColorSettings_Resolution", str(c["color_resolution"]), required=True)
            comp = "ColorCamera"
        else:
            raise ValueError(f"camera.source 는 primary | color : {src}")
        for name in nm.ComponentSelector.symbolics:                      # 2D 영상 하나만 받는다
            nm.ComponentSelector.value = name
            nm.ComponentEnable.value = (name == comp)
        nm.ComponentSelector.value = comp
        if src == "color":
            self._set("PixelFormat", "RGB8")
        for key, node in (("led_power", "LEDPower"), ("exposure_ms", "SinglePatternExposure"),
                          ("color_exposure_ms", "ColorSettings_Exposure"), ("color_iso", "ColorSettings_ISO")):
            if c.get(key) is not None:
                self._set(node, c[key])
        time.sleep(float(c.get("settle_s", 0.5)))                          # 조명/모드 전환 직후 안정 (예제와 같음)
        self.component = comp

    # ------------------------------------------------------------ 찍기
    def _drain(self):
        while True:
            b = self.ia.try_fetch(timeout=0.001)
            if b is None:
                return
            b.queue()

    def _one(self, timeout):
        self.nm.TriggerSoftware.execute()
        with self.ia.fetch(timeout=timeout) as buf:
            p = buf.payload.components[0]
            fmt, w, h = p.data_format, p.width, p.height
            d = p.data.copy()
        if fmt == "RGB8":
            return cv2.cvtColor(d.reshape(h, w, 3), cv2.COLOR_RGB2BGR).astype(np.float32)
        bits = {"Mono8": 8, "Mono10": 10, "Mono12": 12, "Mono16": 16}.get(fmt)
        if bits is None:
            raise RuntimeError(f"지원하지 않는 화소 형식 {fmt}")
        return d.reshape(h, w).astype(np.float32) * (255.0 / ((1 << bits) - 1))

    def grab_raw(self):
        """센서 영상 (8비트 범위 float 평균 → uint8, 흑백이면 3채널)"""
        c = cam_cfg(self.cfg)
        n = max(1, int(c.get("average_frames", 1)))
        timeout = float(c.get("timeout_s", 10))
        gain = float(c.get("intensity_gain", 1.0))
        self._drain()
        acc = None
        for _ in range(n):
            f = self._one(timeout)
            acc = f if acc is None else acc + f
        img = np.clip(acc * (gain / n) + 0.5, 0, 255).astype(np.uint8)
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def grab(self, raw=False):
        """측정용 BGR 영상 (process 적용)"""
        try:
            img = self.grab_raw()
        except Exception as e:                                           # 끊겼으면 한 번 다시 연결
            log(f"촬영 실패 ({e}) → 다시 연결합니다.")
            self.close()
            self.open()
            img = self.grab_raw()
        return process(img, self.cfg, raw)

    @property
    def tag(self):
        return camera_id(self.cfg, self.serial)

    def close(self):
        try:
            if self.ia is not None:
                try:
                    self.ia.stop()
                except Exception:
                    pass
                self.ia.destroy()
        finally:
            self.ia = self.nm = None
            if self.h is not None:
                self.h.reset()
            self.h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def load_cfg(path=None):
    sys.path.insert(0, str(ROOT / "hik"))
    sys.path.insert(0, str(ROOT))
    import common as C
    return C.load_config(path or HERE / "config.yaml")


def main():
    ap = argparse.ArgumentParser(description="Photoneo 2D 촬영")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="연결된 Photoneo 장치")
    g = sub.add_parser("grab", help="1장 찍어 저장")
    g.add_argument("out")
    g.add_argument("--raw", action="store_true", help="돌리기만 (자르기/키우기 없이)")
    b = sub.add_parser("bench", help="연속 촬영 시간 재기")
    b.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.cmd == "list":
        from harvesters.core import Harvester
        with Harvester() as h:
            h.add_file(str(producer_path()), check_existence=True, check_validity=True)
            h.update()
            for d in h.device_info_list:
                p = d.property_dict
                print(f"{p.get('serial_number', ''):<14}{p.get('model', ''):<32}{p.get('vendor', '')}")
        return
    with PhoxiCamera(cfg) as cam:
        if args.cmd == "grab":
            t = time.time()
            img = cam.grab(raw=args.raw)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.out, img)
            print(f"저장: {args.out}  {img.shape[1]}x{img.shape[0]}  ({time.time() - t:.2f} s)  카메라: {cam.tag}")
        else:
            ts = []
            for _ in range(args.n):
                t = time.time()
                cam.grab()
                ts.append(time.time() - t)
            print(f"{args.n}장: 평균 {np.mean(ts):.3f} s, 최소 {min(ts):.3f} s, 최대 {max(ts):.3f} s  (첫 장 {ts[0]:.3f} s)")


if __name__ == "__main__":
    main()
