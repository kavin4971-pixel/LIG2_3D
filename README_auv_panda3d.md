# Panda3D AUV + PPO Viewer (Coriolis update)

이번 업데이트는 **위도(latitude) 기반 코리올리 힘**을 환경 물리로 넣은 버전입니다.

## 파일 역할

- `environment3d.py`
  - 3D 경계 박스
  - `CoriolisConfig`
  - 위도 기반 코리올리 계수 `f = 2 * Ω * sin(latitude)`
  - `coriolis_acceleration()` / `environmental_acceleration()` / `apply_coriolis_to_velocity()`
- `auv_rl_collision_train.py`
  - PPO 학습용 RL 환경
  - AUV와 moving obstacle에 환경 코리올리 효과 반영
  - 벽/장애물 충돌 시 즉시 실패
  - observation에 코리올리 문맥 포함
- `auv_target_sim_panda3d.py`
  - 학습된 PPO policy를 Panda3D에서 재생
  - HUD에 실제 latitude / `f` / 현재 Coriolis 가속도 표시

## 코리올리 모델

현재 기본 모델은 **수평면 f-plane approximation** 입니다.

좌표 해석:

- X: local east-west
- Y: local north-south
- Z: up-down

적용 가속도:

```python
a_c = [f * v_y, -f * v_x, 0.0]
```

여기서

```python
f = 2 * Ω * sin(latitude)
```

입니다.

## RL observation 변경

에이전트가 위도에 따른 편향을 학습할 수 있도록 observation에는 아래 정보가 포함됩니다.

- 현재 속도 벡터 `(vx, vy, vz)`
- 목표 상대 오차 `(target - agent)`
- 목표 거리
- 정규화된 위도 `latitude / 90`
- 정규화된 코리올리 파라미터 `f / (2Ω)`
- 주변 obstacle 상대 정보

즉, 질문에서 말한

- 현재 속도 벡터
- 현재 위도 또는 `f`
- 목표 지점과의 오차

가 모두 observation에 들어갑니다.

## 위도 설정

고정 위도:

```bash
python auv_rl_collision_train.py --mode train --latitude-deg 36.0
```

에피소드마다 위도 랜덤화:

```bash
python auv_rl_collision_train.py --mode train --latitude-range 10 50
```

## Viewer 실행

```bash
python auv_target_sim_panda3d.py --model-path runs/auv_ppo/final_model
```

Viewer에서 고정 위도 override:

```bash
python auv_target_sim_panda3d.py --model-path runs/auv_ppo/final_model --latitude-deg 45
```

Viewer에서 위도 범위 override:

```bash
python auv_target_sim_panda3d.py --model-path runs/auv_ppo/final_model --latitude-range 20 60
```

## 주의

Observation 차원이 바뀌었기 때문에, 이전 observation 구조로 학습한 PPO 체크포인트는 새 코드와 바로 호환되지 않습니다.
새 구조로 다시 학습한 뒤 `auv_target_sim_panda3d.py`에서 불러오는 것을 권장합니다.
