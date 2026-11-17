#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoneo 측정 서버(server.py)에 요청 보내기. 표준 라이브러리만 써서 바로 뜬다.

  python3 photoneo/client.py measure                       # 찍고 측정 → 한글 요약
  python3 photoneo/client.py measure --debug debug.jpg     # 디버그 그림도
  python3 photoneo/client.py measure --save now.png --json # 찍은 사진 저장, 결과는 JSON 으로만
  python3 photoneo/client.py grab out.png [--raw]          # 찍어서 저장만 (--raw: 돌리기만, 자르기/키우기 없이)
  python3 photoneo/client.py reload                        # 보정/설정을 바꾼 뒤 서버에 다시 읽게
  python3 photoneo/client.py ping | stop

다른 파이썬 프로그램에서:
  from photoneo.client import request
  r = request({"cmd": "measure"})        # r["result"]["rotate_cw_deg"]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

DEFAULT = ("127.0.0.1", 5577)


def server_addr():
    """photoneo/config.yaml 의 server (yaml 없이 간단히 읽음)"""
    host, port = DEFAULT
    p = Path(__file__).resolve().parent / "config.yaml"
    try:
        sec = False
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("server:"):
                sec = True
                continue
            if sec and line and not line.startswith(" "):
                break
            if sec:
                k, _, v = line.split("#")[0].strip().partition(":")
                if k == "host" and v.strip():
                    host = v.strip()
                elif k == "port" and v.strip():
                    port = int(v)
    except Exception:
        pass
    return host, port


def request(req, addr=None, timeout=60.0):
    host, port = addr or server_addr()
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        raise RuntimeError(f"측정 서버({host}:{port})에 연결하지 못했습니다: {e}\n"
                           f"먼저 켜세요: python3 photoneo/server.py")
    return json.loads(buf.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Photoneo 측정 서버에 요청")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="찍고 측정")
    m.add_argument("--debug", default=None, help="디버그 그림 저장 경로")
    m.add_argument("--save", default=None, help="찍은 사진 저장 경로")
    m.add_argument("--json", action="store_true", help="결과를 JSON 으로만")
    g = sub.add_parser("grab", help="찍어서 저장만")
    g.add_argument("out")
    g.add_argument("--raw", action="store_true", help="돌리기만 (자르기/키우기 없이)")
    for c in ("reload", "ping", "stop"):
        sub.add_parser(c)
    args = ap.parse_args()

    addr = None
    if args.host or args.port:
        h, p = server_addr()
        addr = (args.host or h, args.port or p)
    req = {"cmd": args.cmd}
    if args.cmd == "measure":
        for k in ("debug", "save"):
            if getattr(args, k):
                req[k] = str(Path(getattr(args, k)).resolve())     # 서버는 다른 폴더에서 돌 수 있어 절대 경로로
    elif args.cmd == "grab":
        req.update(save=str(Path(args.out).resolve()), raw=args.raw)
    try:
        r = request(req, addr)
    except RuntimeError as e:
        sys.exit(str(e))

    if args.cmd == "measure" and args.json:
        print(json.dumps(r.get("result", r), ensure_ascii=False, indent=2))
    elif args.cmd == "measure" and r.get("ok"):
        print(r["summary"])
        if r.get("debug"):
            print(f"  디버그 그림: {r['debug']}")
        if r.get("saved"):
            print(f"  사진: {r['saved']}")
        t = r.get("time", {})
        print(f"  시간: 촬영 {t.get('grab_s', 0):.2f} s + 측정 {t.get('measure_s', 0):.2f} s")
    elif r.get("ok"):
        print(json.dumps({k: v for k, v in r.items() if k != "ok"}, ensure_ascii=False))
    if not r.get("ok"):
        sys.exit(f"실패: {r.get('error')}")


if __name__ == "__main__":
    main()
