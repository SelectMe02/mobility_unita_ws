# planner — 경로 계획

추후 waypoint 로딩과 목표 경로 생성 코드를 배치합니다.

| 경로 | 역할 |
| --- | --- |
| `config/` | 파라미터 YAML, 데이터 및 설정 파일 |
| `launch/` | ROS launch 파일 |
| `scripts/` | 실행 가능한 ROS 노드 |
| `src/` | 재사용할 계산 로직 및 내부 모듈 |
| `test/` | 기능 검증 테스트 |

빈 폴더는 `.gitkeep`으로 Git에 유지합니다. 실제 파일을 추가하면 해당 `.gitkeep`은 삭제할 수 있습니다.

현재는 폴더 구조만 준비되어 있으며 실행 노드·설정·테스트는 아직 없습니다.
노드를 추가하면 `CMakeLists.txt`의 `catkin_install_python`에 명시적으로 등록합니다.
Python 모듈을 만들 때는 `setup.py`와 `catkin_python_setup()`을 추가하고, 의존성은 `package.xml`에도 선언합니다.
