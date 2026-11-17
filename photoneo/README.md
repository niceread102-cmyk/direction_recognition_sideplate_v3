# Photoneo 2D로 side_plate 방향 맞추기

Hikvision 2D 카메라와 **같은 측정 알고리즘**(`../side_plate_notch.py`)을 Photoneo PhoXi 3D Scanner Gen3 (M)으로 찍은 2D 영상에 씁니다.
카메라 접속은 `~/Projects/photoneo-python-examples/GigE-V/harvesters` 예제와 같은 방식(harvesters + ImpactAcquire GenTL)입니다. PhoXi Control은 필요 없습니다.
명령은 모두 상위 폴더(`~/Projects/direction_recognition_sideplate_v2`)에서 실행합니다.

## 빠르게 찍는 방법: 서버를 켜 두기

```bash
python3 photoneo/server.py                         # 켜 두기 (터미널 하나를 이것만 쓰게)
python3 photoneo/client.py measure                 # 측정
python3 photoneo/client.py measure --debug d.jpg   # 디버그 그림도
python3 photoneo/client.py measure --json          # 다른 프로그램 연동용
```

서버는 스캐너 접속(약 5 s)과 STL 준비를 한 번만 하고 계속 열어 둡니다. 측정 한 번에 드는 시간은 다음과 같습니다.

| 단계 | 시간 |
|---|---|
| 촬영 (소프트웨어 트리거, 2D만) | 약 0.2~0.35 s |
| 부품 위치 찾기 | 약 0.6 s |
| 각도 측정 | 약 2.2 s |
| **합계** | **약 3 s** |

서버 없이 매번 접속하면 접속만 5 s가 더 걸립니다.

- 끄기: `python3 photoneo/client.py stop` 또는 서버 터미널에서 Ctrl+C
- 파이썬에서 직접 호출: `from photoneo.client import request; request({"cmd": "measure"})["result"]["rotate_cw_deg"]`
- 터미널을 닫아도 계속 돌게 하려면 `nohup python3 photoneo/server.py > photoneo/server.log 2>&1 &`로 켭니다.

## 2D 촬영 설정 (`photoneo/config.yaml`의 `camera`)

- `CameraOnlyMode`로 패턴을 투사하지 않고 카메라 영상만 찍습니다. 한 장에 약 0.2 s 걸리고, 3D 스캔은 약 0.7 s입니다.
- **흑백(3D 카메라 + LED 조명)이 컬러 카메라보다 낫습니다.** 같은 부품을 같은 크기로 놓고 비교했습니다(`images/compare_mono_vs_color.jpg`).

  | | 노이즈 대비 밝기 범위 | 경계 선명도 |
  |---|---|---|
  | 흑백, 노출 40.96 ms | 76 | 0.125 |
  | 컬러, 노출 5.12 ms | 31 | 0.087 |

  컬러 카메라는 화소가 작고 실내 조명에만 의존해서 각인이 흐리게 나옵니다. 컬러로 바꾸려면 `source: color`로 바꿉니다.
- 노출 40.96 ms, `intensity_gain: 1.4`를 씁니다. 기본 노출 10.24 ms로 찍으면 부품 밝기가 약 43/255로 너무 어둡습니다.
- 영상 처리:
  - `rotate: 180`: 카메라가 거꾸로 달려 있어 돌립니다. 그러면 각도기 0°가 화면 위로 옵니다.
  - `roi`: 부품 주변만 잘라 씁니다.
  - `upscale: 5`: 측정 코드의 px 단위 설정은 원판 반지름 약 600 px에 맞춰져 있습니다. Photoneo에서는 부품이 반지름 약 111 px로 작게 찍혀서, 5배 키워 약 557 px로 맞춥니다.

## 보정

카메라·`roi`·`upscale`·`rotate`를 바꿨을 때 다시 보정합니다. 보정 파일에 이 설정들이 기록되고, 지금 설정과 다르면 측정을 멈춥니다.

```bash
python3 photoneo/calibrate_pho.py click --auto     # 자동 (STL을 크기별로 맞춰 보고 가장 잘 맞는 원)
python3 photoneo/calibrate_pho.py click            # 수동 (테두리 클릭)
```

- 결과는 `photoneo/calib.yaml`, 확인 그림은 `images/calib_click_fit.jpg`입니다.
- 서버가 켜져 있으면 새 보정을 바로 읽습니다.
- 부품을 보정 때와 다른 자리에 놓아도 됩니다. 서버가 측정 전에 부품 위치를 먼저 찾습니다(`matching.locate_part`).

## 문양 쪽 판정: 기준 사진 비교

형상은 180° 대칭이라 문양 쪽은 사진으로 판정합니다. Photoneo 영상에서는 반쪽 평균값 비교의 차이가 3~9%밖에 안 돼서, 자세에 따라 틀렸습니다(`images.test.jpg`). 그래서 기준 사진 비교를 씁니다(`decision.method: reference`).

1. 부품 윗면을 "모델 각도 × 반지름" 띠 영상으로 펼칩니다.
2. 기준 띠 영상(`pattern_ref.npz`)과 그대로 닮았는지, 180° 뒤집어 닮았는지 비교해 큰 쪽을 고릅니다.

격자, "1-4", "25" 문양의 모양 전체를 씁니다. 띠 영상은 띠 통과(4 px 흐림 − 16 px 흐림)로 주물 표면의 반짝임을 지운 뒤 비교합니다. 자세 3개로 기준에 넣지 않은 자세를 맞히는 시험(하나 빼고 시험)에서 모두 맞았고, 가장 작은 닮음 차는 0.165였습니다.

- 닮음 차가 0.08보다 작으면 `문양 판정 불확실`로 내보냅니다(`reference_min_margin`).
- 기준 사진이 없거나 예전 방식이면, 판정을 모두 불확실로 냅니다.

기준 만들기는 보정을 다시 하거나 조명을 바꿨을 때만 합니다.

```bash
python3 photoneo/calibrate_pho.py pattern-ref                       # 지금 1장 찍어서
python3 photoneo/calibrate_pho.py pattern-ref --image a.png b.png   # 여러 자세 사진으로 (더 튼튼)
```

- 첫 사진은 확인 그림(`images/pattern_ref_check.jpg`)을 보고 직접 답합니다. 초록 점(STL 문양)이 각인 격자 위에 있으면 y입니다.
- 나머지 사진은 자동으로 방향을 맞춰 평균합니다.
- 지금 기준은 세 자세(`images/ref_pose1~3.png`)로 만들었습니다. 자세를 더 넣을수록 튼튼해집니다.

## 각도기 기준 각도 보정

측정 각도는 원래 화면 위쪽이 0°입니다. 인쇄한 각도기(`protractor_A4.pdf`)가 화면에 대해 돌아가 있으면 그만큼 차이가 납니다. 2026-11-12에는 1.31° 돌아가 있었습니다. 각도기의 큰 점 12개로 회전을 재서 모든 각도 출력에 반영합니다.

```bash
python3 photoneo/calibrate_pho.py frame      # 각도기가 화면에 다 보이게 놓고 실행
```

- 결과는 `calib.yaml`의 `frame_offset_deg`에 저장됩니다. 확인 그림은 `images/frame_check.jpg`이고, 보라색 선이 각도기 0°입니다.
- 각도기(또는 스테이지)를 카메라에 대해 움직이면 다시 합니다. 측정할 때 각도기가 보정 때보다 0.3° 넘게 돌아가 있으면 경고가 나옵니다.

측정 결과에는 `각도기로 읽으면: 빨간 홈 A …°, B …°`도 함께 나옵니다.

- 부품이 각도기 중심에서 벗어나 있으면, 눈금으로 읽은 값은 홈의 실제 방향과 다릅니다. 3 mm 벗어나면 최대 약 5° 차이가 나고, A와 B가 서로 반대로 틀립니다. 직접 읽은 값은 이 줄과 비교하세요.
- 이 각도기 인쇄는 가로세로가 약 2.2% 다르게 찍혀서, 눈금 자체도 방향에 따라 최대 약 0.6° 틀립니다. 이 줄은 그것까지 계산에 넣은 값입니다.

## 빨간 홈 위치 보정 (각도기 판독값으로)

빨간 홈은 카메라에 보이지 않아서 STL 설계 위치(긴 홀 기준)를 씁니다. 실제 부품의 홈이 설계와 다르면 각도가 일정하게 틀립니다. 그럴 때 각도기로 읽은 값으로 보정합니다.

1. 부품을 각도기 위에 놓습니다. 서버가 켜져 있어야 합니다.
2. A, B 홈을 각도기 눈금으로 읽어서 넣습니다.
3. 자세를 바꿔 가며 3~5번 반복합니다.
4. 평균으로 보정을 적용합니다.

```bash
python3 photoneo/calibrate_pho.py notch --a 83.5 --b 259.0     # 2~3번 반복
python3 photoneo/calibrate_pho.py notch                        # 지금까지 넣은 값과 평균/흩어짐 보기
python3 photoneo/calibrate_pho.py notch --apply                # 평균으로 보정 적용
```

- 넣은 값은 서버가 계산한 "각도기로 읽으면" 값과 비교됩니다. 이 값에는 부품 중심 어긋남과 인쇄 찌그러짐이 이미 들어 있습니다.
- 평균 차가 1°보다 작으면 보정은 필요 없습니다.
- 결과는 `calib.yaml`의 `notch_offset_deg`에 저장됩니다.

## 단독 촬영 (서버 없이)

```bash
python3 photoneo/pho.py list                  # 장치 목록
python3 photoneo/pho.py grab out.png          # 1장 찍기 (돌리기/자르기/키우기 적용)
python3 photoneo/pho.py grab out.png --raw    # 돌리기만 (ROI 정할 때)
python3 photoneo/pho.py bench -n 10           # 연속 촬영 시간
```

## 파일

| 파일 | 내용 |
|---|---|
| `pho.py` | Photoneo 2D 촬영 (한 번 열어 두고 계속 찍기, 돌리기/자르기/키우기) |
| `server.py` | 상주 서버 (카메라 + 측정 준비를 열어 둠, 부품 위치 찾기) |
| `client.py` | 서버에 요청 (표준 라이브러리만 사용) |
| `calibrate_pho.py` | 보정 (자동/클릭, STL 다듬기는 `../calibrate.py`의 refine) |
| `config.yaml` | 설정. 알고리즘 설정은 `../config.yaml`과 같고 카메라/보정/서버만 다름 |
| `calib.yaml` | 보정 결과 (Hikvision 보정과 따로) |
