# Panda3D PPO Policy Viewer for AUV Navigation

이 버전의 `auv_target_sim_panda3d.py`는 더 이상 hand-crafted seek guidance를 쓰지 않습니다.
대신 `auv_rl_collision_train.py`의 `AUVNavigationRLEnv`를 그대로 불러와서,
**학습된 PPO policy가 Panda3D 화면에서 실제로 움직이도록** 연결한 viewer 입니다.

핵심 아이디어는 간단합니다.

- `auv_rl_collision_train.py`가 만든 **동일한 상태/관측/충돌 규칙**을 그대로 사용
- Stable-Baselines3 `PPO.load()`로 저장된 정책 로드
- 매 simulation step 마다 `model.predict(obs, deterministic=True)`로 action 추론
- `env.step(action)` 결과를 Panda3D 노드 위치에 반영

즉, **학습 환경과 시각화 환경이 분리되지 않고 동일한 dynamics를 공유**합니다.

## 필요한 파일

같은 폴더에 아래 파일이 있어야 합니다.

- `auv_target_sim_panda3d.py`
- `auv_rl_collision_train.py`
- `environment3d.py`
- 학습된 모델 파일 (`runs/auv_ppo/final_model.zip` 같은 형태)
- 가능하면 같은 폴더/디렉터리에 저장된 `training_config.json`

## 설치

```bash
pip install panda3d==1.10.16 numpy
pip install 'stable-baselines3[extra]'
```

## 실행

기본 모델 경로(`runs/auv_ppo/final_model`) 사용:

```bash
python auv_target_sim_panda3d.py
```

명시적으로 모델 지정:

```bash
python auv_target_sim_panda3d.py --model-path runs/auv_ppo/final_model
```

모델 옆에 있는 `training_config.json`을 자동으로 읽지 못했을 때 직접 지정:

```bash
python auv_target_sim_panda3d.py \
  --model-path runs/auv_ppo/final_model \
  --config-path runs/auv_ppo/training_config.json
```

stochastic inference로 보기:

```bash
python auv_target_sim_panda3d.py --stochastic
```

충돌/성공 후 자동 리셋 끄기:

```bash
python auv_target_sim_panda3d.py --auto-reset-delay -1
```

## 조작키

- `SPACE` : pause / resume
- `R` : 현재 episode 즉시 reset
- `D` : deterministic ↔ stochastic inference 토글
- `C` : 카메라 리셋
- `ESC` : 종료

## HUD에서 보는 항목

- 현재 상태 (`RUNNING`, `SUCCESS`, `FAILED (OBSTACLE COLLISION)` 등)
- 에피소드 번호와 성공/실패/타임아웃 누적 수
- PPO inference 모드 (deterministic / stochastic)
- AUV 위치, Target 위치, 거리, 속도
- 마지막 action / reward
- 장애물 개수, 이동 속도, 복잡도
- 최소 clearance

## 충돌 규칙

이 viewer는 학습용 RL 환경의 규칙을 그대로 따릅니다.

- 벽 충돌 = 즉시 실패
- 장애물 충돌 = 즉시 실패
- timeout = episode 종료
- target capture radius 안으로 들어가면 성공

즉, 이전처럼 충돌 후 조금 밀려나서 계속 가는 방식이 아니라,
**충돌이 발생하는 순간 episode가 끝나는 hard-failure 방식**입니다.

## 중요한 점

정책이 잘 움직이려면 **viewer의 환경 설정이 학습 때와 같아야 합니다.**
그래서 `training_config.json`을 자동 로드하도록 해 두었습니다.

- `training_config.json`이 있으면 그 설정을 우선 사용
- 없으면 `auv_rl_collision_train.py`의 기본 `LayoutConfig()` / `AUVSimConfig()` 사용

학습할 때 obstacle count, complexity, observation size, max speed 등을 바꿨다면,
반드시 같은 config를 viewer에도 적용해야 합니다.
