# Panda3D AUV Target Simulation

이 예제는 `environment3d.py`에 **환경 관련 설정을 집중**시키고,
`auv_target_sim_panda3d.py`에는 **AUV 동작과 Panda3D 런타임 로직만** 남기도록 정리한 버전입니다.

## 파일 구성

- `environment3d.py`
  - 3D 환경 경계
  - 구형 장애물
  - 장애물 복잡도 계산
  - 시작 위치 / 타깃 샘플링 규칙
  - 환경 기본 설정(`make_default_environment_config()`)
- `auv_target_sim_panda3d.py`
  - Panda3D 기반 AUV 시뮬레이션 메인 스크립트
  - AUV 속도, 가속도, capture radius 등 AUV 관련 파라미터
  - 렌더링 / 카메라 / HUD / update loop

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

## 이번 리팩터링의 핵심

이전에는 아래 값들이 `auv_target_sim_panda3d.py`에 직접 들어 있었습니다.

- 환경 크기
- 환경 원점
- 장애물 반지름
- 장애물 개수 / 복잡도
- 장애물 생성 여유 거리
- 시작 위치 생성 규칙
- 타깃 샘플링 규칙
- 환경 바닥 grid 간격

이제 이 값들은 모두 `environment3d.py`의
`make_default_environment_config()` 안에서 관리합니다.

즉, **환경을 바꾸고 싶으면 environment3d.py만 수정하면 됩니다.**

## 환경 설정을 바꾸는 위치

`environment3d.py`의 아래 함수만 보면 됩니다.

```python
def make_default_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        size=(30.0, 30.0, 12.0),
        origin=(-15.0, -15.0, 0.0),
        obstacle_field=ObstacleFieldConfig(
            radius=1.35,
            count=None,
            complexity=0.015,
            clearance_from_auv_multiplier=2.0,
            clearance_margin=0.05,
            max_attempts_per_obstacle=1200,
        ),
        start=StartPointConfig(
            x_margin=2.0,
            y_margin=2.0,
            z_fraction=0.5,
            reserved_clearance=1.75,
        ),
        target=TargetPointConfig(
            boundary_clearance_base=0.35,
            boundary_clearance_min=0.80,
            obstacle_clearance_base=0.15,
            min_distance_from_current=6.0,
            max_attempts=2000,
            fallback_min_z=0.80,
            fallback_extra_push=0.05,
        ),
        visuals=EnvironmentVisualConfig(
            wire_box_thickness=2.0,
            floor_grid_spacing=2.0,
        ),
    )
```

## 장애물 복잡도 정의

장애물 복잡도는 아래처럼 정의합니다.

```python
obstacle_complexity = total_obstacle_volume / environment_volume
```

구형 장애물 반지름이 `r`일 때,
구 하나의 부피는 다음과 같습니다.

```python
sphere_volume = (4.0 / 3.0) * math.pi * (r ** 3)
```

복잡도로부터 장애물 개수는 대략 아래처럼 결정됩니다.

```python
count ≈ round((obstacle_complexity * environment_volume) / sphere_volume)
```

실제 배치에서는 장애물끼리 겹치지 않아야 하고,
시작 위치와도 충분한 간격을 둬야 하므로,
**실제 배치 수는 요청 수보다 적을 수 있습니다.**

## 장애물 설정 예시

### 1) 수동 개수 모드

```python
obstacle_field=ObstacleFieldConfig(
    radius=1.20,
    count=12,
    complexity=None,
)
```

### 2) 복잡도 기반 자동 모드

```python
obstacle_field=ObstacleFieldConfig(
    radius=1.35,
    count=None,
    complexity=0.015,
)
```

## AUV 쪽에서 남는 설정

`auv_target_sim_panda3d.py`에는 이제 주로 아래만 남습니다.

```python
self.capture_radius = 0.50
self.max_speed = 6.0
self.max_accel = 7.5
self.slowdown_radius = 7.0
self.auv_radius = 0.65
self.target_radius = 0.42
```

즉,

- 환경을 바꾸는 일 → `environment3d.py`
- AUV 성능/운동을 바꾸는 일 → `auv_target_sim_panda3d.py`

로 역할이 분리됩니다.

## 다음 단계 추천

1. 장애물 회피 벡터(avoidance force) 추가
2. 단순 seek 대신 PID 또는 LOS guidance 적용
3. 수중 드래그/부력/유체저항 모델 추가
4. 센서(sonar/FOV)와 waypoint 기반 경로 계획 추가
