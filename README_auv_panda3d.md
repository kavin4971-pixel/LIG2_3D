# Panda3D AUV Target Simulation

이 예제는 `environment3d.py`에 **환경 관련 설정**을 모으고,
`auv_target_sim_panda3d.py`에는 **AUV 동작과 Panda3D 런타임 로직**만 남기도록 정리한 버전입니다.

## 파일 구성

- `environment3d.py`
  - 3D 환경 경계
  - 구형 장애물 생성
  - 장애물 복잡도 계산
  - 시작 위치 / 타깃 샘플링 규칙
  - **동적 장애물 이동 설정**
    - 랜덤 초기 방향
    - 속도 = `AUV 최대 속도 × 비율`
    - 벽 / 다른 장애물 / Target 주변 보호영역 충돌 시 랜덤 bounce
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

## 현재 동작

- AUV는 target을 향해 이동합니다.
- 장애물은 각자 랜덤한 초기 방향으로 계속 이동합니다.
- 장애물 속도는 아래처럼 결정됩니다.

```python
obstacle_speed = obstacle_motion.speed_ratio_to_auv_max * auv_max_speed
```

기본값은 `0.10` 이므로,
AUV 최대 속도가 `6.0 m/s`이면 장애물은 `0.6 m/s`로 움직입니다.

## 장애물 bounce 조건

장애물은 다음 경우에 현재 진행 방향을 버리고,
**충돌 면의 바깥쪽 반구 안에서 랜덤한 새 방향**을 선택합니다.

- 환경 벽에 닿을 때
- 다른 장애물과 겹칠 때
- Target 주변 보호영역에 닿을 때

Target 보호영역 반경은 기본적으로 아래와 같습니다.

```python
target_keepout_radius = target_radius + 1.0
```

즉, 장애물 중심은 항상
`obstacle_radius + target_keepout_radius` 이상 떨어지도록 밀려납니다.

## 환경 설정 위치

환경을 바꾸고 싶다면 `environment3d.py`의
`DEFAULT_ENVIRONMENT_CONFIG`만 수정하면 됩니다.

예시:

```python
DEFAULT_ENVIRONMENT_CONFIG = EnvironmentConfig(
    size=(30.0, 30.0, 12.0),
    origin=(-15.0, -15.0, 0.0),
    random_seed=7,
    obstacle=ObstacleConfig(
        radius=1.35,
        count=None,
        complexity=0.015,
        clearance_multiplier=2.0,
        clearance_padding=0.05,
        max_attempts_per_obstacle=1200,
        motion=ObstacleMotionConfig(
            enabled=True,
            speed_ratio_to_auv_max=0.10,
            target_clearance=1.0,
            resolution_passes=2,
        ),
    ),
    spawn=SpawnConfig(
        start_offset_xy=(2.0, 2.0),
        start_height_ratio=0.5,
        start_reserved_extra=1.75,
        target_boundary_padding=0.35,
        min_target_boundary_clearance=0.80,
        target_obstacle_padding=0.15,
        target_min_distance_from_agent=6.0,
        target_max_attempts=2000,
    ),
    visual=EnvironmentVisualConfig(
        box_line_thickness=2.0,
        grid_spacing=2.0,
    ),
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

- 환경 / 장애물 / obstacle motion 변경 → `environment3d.py`
- AUV 성능 / guidance 변경 → `auv_target_sim_panda3d.py`

로 역할이 분리됩니다.

## 다음 단계 추천

1. 장애물 회피 벡터(avoidance force) 추가
2. 단순 seek 대신 PID 또는 LOS guidance 적용
3. 수중 드래그/부력/유체저항 모델 추가
4. 센서(sonar/FOV)와 waypoint 기반 경로 계획 추가
