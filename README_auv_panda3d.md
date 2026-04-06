# Panda3D AUV Target Simulation

이 예제는 업로드한 `environment3d.py`의 `Environment3D` 클래스를 그대로 사용해서
3차원 경계 박스 내부에서 AUV가 Target으로 이동하는 시뮬레이션을 보여줍니다.

## 파일 구성

- `environment3d.py` : 사용자가 업로드한 3D 환경 경계 클래스
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

## 핵심 아이디어

- `Environment3D`의 `min_bound`, `max_bound`를 Panda3D 씬 경계로 사용
- AUV는 현재 위치에서 타깃까지의 방향 벡터를 따라 가속
- `max_speed`, `max_accel`, `slowdown_radius`로 움직임 튜닝
- `clamp()`를 사용해 경계 밖으로 나가지 않게 제한
- Panda3D task 루프에서 매 프레임 상태 업데이트

## 크기 변경

현재 예제는 아래처럼 환경을 설정합니다.

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

1. 장애물(sphere / box) 추가
2. 단순 seek 대신 PID 또는 LOS guidance 적용
3. 수중 드래그/부력/유체저항 모델 추가
4. 센서(sonar/FOV)와 waypoint 기반 경로 계획 추가
5. glTF AUV 모델 로딩으로 외형 개선
