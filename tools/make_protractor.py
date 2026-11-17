#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정확도 검증용 인쇄 각도기 (A4, 실제 크기)
    python3 tools/make_protractor.py [출력.pdf]        # 기본: protractor_A4.pdf (이 폴더의 상위)
- 각도: 12시 = 0°, 시계방향 + (side_plate_notch.py 출력과 같은 기준)
- 부품 놓는 원 = STL 바깥 지름, 눈금은 그보다 바깥(부품 테두리와 겹치지 않게)
"""
import math
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "protractor_A4.pdf"

PART_R = 27.83          # STL 바깥 반지름 (mm)
R_TICK = 35.0           # 눈금 시작 반지름 (부품 테두리에서 약 7 mm 바깥)
L1, L5, L10 = 3.0, 5.0, 8.0
R_LABEL = R_TICK + L10 + 4.0

FONT, FONT_B = "Nanum", "NanumB"
pdfmetrics.registerFont(TTFont(FONT, "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont(FONT_B, "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))

W, H = A4
c = canvas.Canvas(str(OUT), pagesize=A4)
c.setTitle("side_plate 검증용 각도기")
CX, CY = W / 2, H - 150 * mm           # 각도기 중심


def pt(r_mm, a_deg):
    """반지름(mm), 사용자 각도(12시=0, 시계방향+) → 페이지 좌표"""
    a = math.radians(a_deg)
    return CX + r_mm * mm * math.sin(a), CY + r_mm * mm * math.cos(a)


# ---------------- 안내문
c.setFont(FONT_B, 15)
c.drawString(18 * mm, H - 18 * mm, "side_plate 정확도 검증용 각도기")
c.setFont(FONT, 9)
lines = [
    "1. '실제 크기(100%)'로 인쇄하세요. '페이지에 맞춤'은 끕니다. 인쇄 후 아래 100 mm 자를 자로 재서 확인하세요.",
    "2. 종이를 카메라 아래에 테이프로 고정하고, 부품을 점선 원 안에 맞춰 올립니다.",
    "3. 아래 바늘 하나를 오려 부품 윗면 테두리에 붙이고, 끝이 눈금에 닿게 합니다. (12시=0°, 시계방향 +)",
    "4. 바늘을 0°에 맞춘 뒤  python3 validate.py capture --out images/val --step 30 --repeat 2  를 실행하고,",
    "   안내되는 각도(빨간 점: 30°마다)로 시계방향으로 돌려 가며 Enter. 부품 중심이 점선 원에서 벗어나지 않게 하세요.",
]
for i, s in enumerate(lines):
    c.drawString(18 * mm, H - (26 + 5 * i) * mm, s)

# ---------------- 부품 자리
c.setLineWidth(0.6)
c.setStrokeGray(0.45)
c.setDash(3, 2)
c.circle(CX, CY, PART_R * mm)
c.setDash()
c.setLineWidth(0.3)
c.line(CX - 6 * mm, CY, CX + 6 * mm, CY)
c.line(CX, CY - 6 * mm, CX, CY + 6 * mm)
c.circle(CX, CY, 1.0 * mm)
c.setFont(FONT, 7)
c.setFillGray(0.4)
c.drawCentredString(CX, CY - PART_R * mm + 3.5 * mm, f"부품 자리 (지름 {2 * PART_R:.1f} mm)")
c.setFillGray(0)

# ---------------- 눈금
c.setStrokeGray(0)
for a in range(360):
    L = L10 if a % 10 == 0 else (L5 if a % 5 == 0 else L1)
    c.setLineWidth(0.45 if a % 10 == 0 else 0.25)
    c.line(*pt(R_TICK, a), *pt(R_TICK + L, a))
c.setLineWidth(0.3)
c.circle(CX, CY, R_TICK * mm)

for a in range(0, 360, 10):
    x, y = pt(R_LABEL, a)
    c.saveState()
    c.translate(x, y)
    c.rotate(-a)                          # 글자 위쪽이 바깥을 향하게
    bold = a % 30 == 0
    c.setFont(FONT_B if bold else FONT, 8.5 if bold else 7)
    c.drawCentredString(0, -1.2 * mm, f"{a}")
    c.restoreState()

# 30° 검증 자리 표시 (빨간 점)
c.setFillColorRGB(0.85, 0.15, 0.15)
for a in range(0, 360, 30):
    c.circle(*pt(R_LABEL + 5.0, a), 1.1 * mm, stroke=0, fill=1)
c.setFillGray(0)

# 0° 위에 화살표 + 회전 방향
x0, y0 = pt(R_LABEL + 9, 0)
c.setFont(FONT_B, 9)
c.drawCentredString(x0, y0, "0° (12시)")
c.setFont(FONT, 8)
c.setLineWidth(0.6)
p = c.beginPath()
r_arc = R_LABEL + 12
p.arc(CX - r_arc * mm, CY - r_arc * mm, CX + r_arc * mm, CY + r_arc * mm, startAng=75, extent=-50)
c.drawPath(p)
xa, ya = pt(r_arc, 65)
c.drawString(xa + 2 * mm, ya, "시계방향 +")
ah = pt(r_arc, 65)
c.line(ah[0], ah[1], ah[0] - 2.2 * mm, ah[1] + 0.6 * mm)
c.line(ah[0], ah[1], ah[0] - 0.4 * mm, ah[1] + 2.2 * mm)

# ---------------- 100 mm 자 (인쇄 배율 확인)
yb = 52 * mm
xb = (W - 100 * mm) / 2
c.setLineWidth(0.5)
c.line(xb, yb, xb + 100 * mm, yb)
for i in range(0, 101):
    L = 4 if i % 10 == 0 else (2.5 if i % 5 == 0 else 1.5)
    c.setLineWidth(0.4 if i % 10 == 0 else 0.2)
    c.line(xb + i * mm, yb, xb + i * mm, yb + L * mm)
    if i % 10 == 0:
        c.setFont(FONT, 6)
        c.drawCentredString(xb + i * mm, yb + 5 * mm, str(i))
c.setFont(FONT_B, 8)
c.drawCentredString(W / 2, yb - 5 * mm, "100 mm (자로 재서 정확히 100 mm 인지 확인)")

# ---------------- 오려 쓰는 바늘 (3개)
c.setFont(FONT, 8)
c.drawString(18 * mm, 36 * mm, "바늘 (오려서 사용, 넓은 쪽을 부품 윗면에 붙이고 뾰족한 끝을 눈금에):")
for k in range(3):
    bx, by = 25 * mm + k * 60 * mm, 22 * mm
    Lb, Wb = 40 * mm, 5 * mm
    p = c.beginPath()
    p.moveTo(bx, by - Wb / 2)
    p.lineTo(bx + Lb - 8 * mm, by - Wb / 2)
    p.lineTo(bx + Lb, by)
    p.lineTo(bx + Lb - 8 * mm, by + Wb / 2)
    p.lineTo(bx, by + Wb / 2)
    p.close()
    c.setLineWidth(0.4)
    c.setDash(2, 1.5)
    c.drawPath(p)
    c.setDash()
    c.setLineWidth(0.3)
    c.line(bx + 2 * mm, by, bx + Lb, by)           # 가운데 기준선

c.setFont(FONT, 6.5)
c.setFillGray(0.4)
c.drawRightString(W - 12 * mm, 10 * mm, "direction_recognition_sideplate_v2 / tools/make_protractor.py")
c.showPage()
c.save()
print(f"저장: {OUT}")
