#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hikvision 산업용 흑백 카메라(MV-CH120-10GM, GigE) 촬영. config.yaml 의 camera 설정을 따른다.
MVS SDK(/opt/MVS) 의 Python 바인딩(MvImport)을 쓴다.

  camera:
    serial: auto          # auto(처음 찾은 카메라) | 시리얼 번호(예: DB2522123)
    pixel_format: Mono8   # Mono8 | Mono10 | Mono12  (10/12 비트는 8비트로 줄여서 넘긴다)
    exposure_us: 20000    # 노출 시간(µs). null 이면 카메라 현재 값 유지
    gain_db: 0            # 게인(dB). null 이면 현재 값 유지
    roi: null             # [x, y, w, h] 카메라에서 잘라 받기. 바꾸면 다시 보정

흑백 영상은 3채널(B=G=R)로 바꿔서 돌려준다. (측정 코드는 BGR 영상을 받는다)
보정 파일에는 어떤 카메라·ROI 로 보정했는지(camera: ...) 기록하고, 촬영할 때 지금 카메라와 다르면 멈춘다.
"""
import os
import platform
import sys
from ctypes import POINTER, byref, cast, memset, sizeof
from pathlib import Path

import cv2
import numpy as np
import yaml

_warned = set()
_sdk = None


def _warn(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"[hik] {msg}", file=sys.stderr)


def sdk(cfg=None):
    """MvImport 모듈(MvCameraControl_class) 을 불러온다. 경로: camera.sdk_path(auto 면 /opt/MVS)"""
    global _sdk
    if _sdk is not None:
        return _sdk
    root = str((cfg or {}).get("camera", {}).get("sdk_path", "auto"))
    root = Path("/opt/MVS" if root == "auto" else root)
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(root / "lib"))    # SDK 가 이 변수로 .so 를 찾는다
    imp = root / "Samples" / platform.machine() / "Python" / "MvImport"
    if not imp.exists():
        imp = root / "Samples" / "64" / "Python" / "MvImport"              # x86_64 설치는 64 폴더
    sys.path.append(str(imp))
    try:
        import MvCameraControl_class as m
    except (ImportError, OSError) as e:
        raise RuntimeError(f"MVS SDK 를 불러오지 못했습니다 ({imp}): {e}. MVS 설치와 camera.sdk_path 를 확인하세요.")
    _sdk = m
    return m


def _s(arr):
    return bytes(arr).split(b"\0")[0].decode("latin-1").strip()


def _check(ret, what):
    if ret != 0:
        m = sdk()
        if ret == m.MV_E_ACCESS_DENIED:
            raise RuntimeError(f"{what} 실패: 카메라를 다른 프로그램(MVS 클라이언트 등)이 쓰고 있습니다. 닫고 다시 실행하세요.")
        raise RuntimeError(f"{what} 실패 (오류 0x{ret:08x})")


def find_device(cfg):
    """config 의 serial 에 맞는 장치 정보와 (모델, 시리얼)"""
    m = sdk(cfg)
    want = str(cfg.get("camera", {}).get("serial", "auto"))
    dl = m.MV_CC_DEVICE_INFO_LIST()
    _check(m.MvCamera.MV_CC_EnumDevices(m.MV_GIGE_DEVICE | m.MV_USB_DEVICE, dl), "카메라 찾기")
    found = []
    for i in range(dl.nDeviceNum):
        d = cast(dl.pDeviceInfo[i], POINTER(m.MV_CC_DEVICE_INFO)).contents
        info = d.SpecialInfo.stGigEInfo if d.nTLayerType == m.MV_GIGE_DEVICE else d.SpecialInfo.stUsb3VInfo
        model, serial = _s(info.chModelName), _s(info.chSerialNumber)
        found.append(f"{model}({serial})")
        if want.lower() == "auto" or want == serial:
            return d, model, serial
    raise RuntimeError(f"카메라 serial={want} 를 찾지 못했습니다. 찾은 장치: {', '.join(found) or '없음'} "
                       f"(GigE 는 PC 랜카드와 같은 대역의 IP 인지 MVS 로 확인)")


def roi_of(cfg):
    roi = cfg.get("camera", {}).get("roi")
    return None if not roi else [int(v) for v in roi]


def camera_id(cfg, model=None, serial=None):
    """보정 파일에 남길 카메라 식별자: hik:<모델>:<시리얼>[:roi=x,y,w,h]"""
    if model is None:
        _, model, serial = find_device(cfg)
    roi = roi_of(cfg)
    return f"hik:{model}:{serial}" + (f":roi={','.join(map(str, roi))}" if roi else "")


def _set(cam, kind, key, val):
    """설정 실패는 경고만 (기종마다 없는 항목이 있다)"""
    f = {"enum": cam.MV_CC_SetEnumValueByString, "float": cam.MV_CC_SetFloatValue,
         "int": cam.MV_CC_SetIntValue, "bool": cam.MV_CC_SetBoolValue}[kind]
    ret = f(key, val)
    if ret != 0:
        _warn(f"{key}={val} 설정 실패 (오류 0x{ret:08x})")
    return ret == 0


def configure(cam, dev, cfg):
    m = sdk()
    c = cfg.get("camera", {})
    if dev.nTLayerType == m.MV_GIGE_DEVICE:
        ps = cam.MV_CC_GetOptimalPacketSize()                 # 실패하면 오류 코드(큰 값)를 돌려준다
        if 0 < ps < 65536:
            _set(cam, "int", "GevSCPSPacketSize", ps)
        else:
            _warn("최적 패킷 크기를 얻지 못했습니다. 랜카드 MTU(점보 프레임)를 확인하세요.")
    _set(cam, "enum", "TriggerMode", "Off")
    _set(cam, "enum", "PixelFormat", str(c.get("pixel_format", "Mono8")))
    roi = roi_of(cfg)
    if roi:
        # 크기를 먼저 줄여야 오프셋을 옮길 수 있다 (오프셋 + 크기 <= 센서 크기)
        _set(cam, "int", "OffsetX", 0)
        _set(cam, "int", "OffsetY", 0)
        for key, v in (("Width", roi[2]), ("Height", roi[3]), ("OffsetX", roi[0]), ("OffsetY", roi[1])):
            if not _set(cam, "int", key, v):
                raise RuntimeError(f"camera.roi {roi} 를 적용하지 못했습니다. 값이 센서 크기 안이고 "
                                   f"카메라 간격(보통 8 또는 16의 배수)에 맞는지 확인하세요.")
    if c.get("exposure_us") is not None:
        _set(cam, "enum", "ExposureAuto", "Off")
        _set(cam, "float", "ExposureTime", float(c["exposure_us"]))
    if c.get("gain_db") is not None:
        _set(cam, "enum", "GainAuto", "Off")
        _set(cam, "float", "Gain", float(c["gain_db"]))
    if c.get("gamma") is not None:
        _set(cam, "bool", "GammaEnable", True)
        _set(cam, "float", "Gamma", float(c["gamma"]))


def _frame_to_array(fr):
    """MV_FRAME_OUT → 2차원 배열 (Mono8: uint8, Mono10/12: 8비트 범위의 float32)"""
    m = sdk()
    fi = fr.stFrameInfo
    h, w, pt = fi.nHeight, fi.nWidth, fi.enPixelType
    if pt == m.PixelType_Gvsp_Mono8:
        return np.ctypeslib.as_array(fr.pBufAddr, shape=(h * w,)).reshape(h, w).copy()
    bits = {m.PixelType_Gvsp_Mono10: 10, m.PixelType_Gvsp_Mono12: 12}.get(pt)
    if bits is None:
        raise RuntimeError(f"지원하지 않는 화소 형식 0x{pt:08x}. camera.pixel_format 을 Mono8/Mono10/Mono12 로 (Packed 제외)")
    raw = np.ctypeslib.as_array(fr.pBufAddr, shape=(h * w * 2,)).view(np.uint16).reshape(h, w)
    return raw.astype(np.float32) * (255.0 / ((1 << bits) - 1))


def grab(cfg, check=True):
    """여러 장 평균한 BGR 한 장 (B=G=R). check 면 보정 파일의 카메라와 대조"""
    m = sdk(cfg)
    c = cfg.setdefault("camera", {})
    m.MvCamera.MV_CC_Initialize()
    cam = m.MvCamera()
    opened = grabbing = False
    try:
        dev, model, serial = find_device(cfg)
        _check(cam.MV_CC_CreateHandle(dev), "카메라 핸들 만들기")
        _check(cam.MV_CC_OpenDevice(m.MV_ACCESS_Exclusive, 0), "카메라 열기")
        opened = True
        configure(cam, dev, cfg)
        _check(cam.MV_CC_StartGrabbing(), "촬영 시작")
        grabbing = True

        fr = m.MV_FRAME_OUT()
        memset(byref(fr), 0, sizeof(fr))
        timeout = int(c.get("timeout_ms", 3000))
        n_warm, n = int(c.get("warmup_frames", 3)), int(c.get("average_frames", 5))
        acc, k, lost = None, 0, 0
        while k < n_warm + n:
            ret = cam.MV_CC_GetImageBuffer(fr, timeout)
            if ret != 0:
                raise RuntimeError(f"프레임을 받지 못했습니다 (오류 0x{ret:08x}). 랜 연결/패킷 크기/노출 시간을 확인하세요.")
            try:
                miss = int(fr.stFrameInfo.nLostPacket)
                f = _frame_to_array(fr) if k >= n_warm and miss == 0 else None
            finally:
                cam.MV_CC_FreeImageBuffer(fr)
            if miss:                                         # 패킷이 빠진 프레임은 일부가 검게 나와 버린다
                lost += 1
                if lost > 3 * n:
                    raise RuntimeError("패킷 손실이 계속됩니다. 점보 프레임(MTU 9000)과 랜 케이블을 확인하세요.")
                continue
            if f is not None:
                acc = f.astype(np.float32) if acc is None else acc + f
            k += 1
        if lost:
            _warn(f"패킷이 빠진 프레임 {lost}장을 버렸습니다.")
        gray = np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)
        _warn(f"{model} ({serial})  {gray.shape[1]}x{gray.shape[0]}  {c.get('pixel_format', 'Mono8')}  "
              f"{n}장 평균")
    finally:
        if grabbing:
            cam.MV_CC_StopGrabbing()
        if opened:
            cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        m.MvCamera.MV_CC_Finalize()
    c["_id"] = camera_id(cfg, model, serial)                 # calibrate_plane.py 가 보정 파일에 남긴다
    if check:
        check_calib(cfg, c["_id"])
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def check_calib(cfg, cam):
    """보정 파일이 이 카메라(+ROI)로 만든 것인지 확인 (다르면 멈춤, 기록이 없으면 경고)"""
    p = Path(cfg["calibration"]["file"])
    if not p.is_absolute():
        p = Path(cfg.get("_dir", ".")) / p
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        tag = (yaml.safe_load(f) or {}).get("camera")
    if tag is None:
        _warn(f"{p.name} 에 카메라 기록이 없습니다. calibrate.py --camera 로 이 카메라를 보정하세요.")
    elif tag != cam:
        sys.exit(f"{p} 는 다른 카메라(또는 다른 ROI)로 보정한 파일입니다.\n  보정: {tag}\n  지금: {cam}\n"
                 f"calibrate.py --camera 로 다시 보정하세요.")
