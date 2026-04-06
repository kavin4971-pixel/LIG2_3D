# Panda3D AUV Target Simulation

이 예제는 업로드한 `environment3d.py`의 `Environment3D`를 확장해서,
3차원 경계 박스 내부에서 AUV가 Target으로 이동하고,
환경 안에 **구형(sphere) 장애물**이 존재하는 시뮬레이션을 보여줍니다.

## 파일 구성

- `environment3d.py` : 3D 환경 경계 + 구형 장애물 + 복잡도 계산
- `auv_target_sim_panda3d.py` : Panda3D 기반 AUV 시뮬레이션 메인 스크립트

## 설치

```bash
pip install panda3d==1.10.16 numpy
```

## 실행

두 파일을 같은 폴더에 둔 뒤:

```bash
python auv_target_sim_panda3d.py
```

## 조작키

- `SPACE` : 일시정지 / 재개
- `R` : 타깃 위치 랜덤 재생성
- `C` : 카메라 리셋
- `ESC` : 종료

## 이번 단계에서 추가된 내용

- 환경 내부에 **구형 장애물** 랜덤 생성
- 장애물 파라미터:
  - `obstacle_radius`
  - `obstacle_count`
  - `obstacle_complexity`
- AUV 시작 위치와 타깃 위치가 장애물과 겹치지 않도록 샘플링
- AUV가 장애물 내부로 들어가면 표면 바깥으로 밀어내는 간단한 충돌 처리
- HUD에 장애물 개수 / 실제 복잡도 표시

## 장애물 복잡도 정의

장애물 복잡도는 아래처럼 정의했습니다.

```python
obstacle_complexity = total_obstacle_volume / environment_volume
```

구형 장애물 반지름이 `r`일 때,
구 하나의 부피는 다음과 같습니다.

```python
sphere_volume = (4.0 / 3.0) * math.pi * (r ** 3)
```

따라서 복잡도로부터 장애물 개수는 대략 아래처럼 결정됩니다.

```python
count ≈ round((obstacle_complexity * environment_volume) / sphere_volume)
```

실제 배치에서는 장애물끼리 겹치지 않아야 하고,
벽/시작 위치와도 여유 간격을 둬야 하므로,
**요청한 개수보다 실제 배치 개수가 더 적을 수 있습니다.**
HUD에는 `실제 배치 수 / 요청 수`가 함께 표시됩니다.

## 장애물 설정 방법

`auv_target_sim_panda3d.py` 안에서 아래 블록을 조절하면 됩니다.

```python
self.obstacle_radius = 1.35
self.obstacle_count: int | None = None       # 정수로 넣으면 수동 개수 모드
self.obstacle_complexity: float | None = 0.015
self.obstacle_clearance = 2.0 * self.auv_radius + 0.05
```

### 1) 수동 개수 모드

```python
self.obstacle_radius = 1.20
self.obstacle_count = 12
self.obstacle_complexity = None
```

### 2) 복잡도 기반 자동 모드

```python
self.obstacle_radius = 1.35
self.obstacle_count = None
self.obstacle_complexity = 0.015
```

## 현재 예제의 환경 크기

```python
self.env = Environment3D.from_size(
    size=(30.0, 30.0, 12.0),
    origin=(-15.0, -15.0, 0.0),
)
```

원하면 다음처럼 바꿀 수 있습니다.

```python
self.env = Environment3D()
```

그러면 업로드한 기본 환경인 `(0,0,0) ~ (3,3,3)` 큐브를 그대로 사용합니다.

## 다음 단계 추천

1. 장애물 회피 벡터(avoidance force) 추가
2. 단순 seek 대신 PID 또는 LOS guidance 적용
3. 수중 드래그/부력/유체저항 모델 추가
4. 센서(sonar/FOV)와 waypoint 기반 경로 계획 추가
5. glTF AUV 모델 로딩으로 외형 개선
