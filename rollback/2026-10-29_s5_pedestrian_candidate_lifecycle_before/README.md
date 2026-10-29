# S5 pedestrian candidate lifecycle

Changes: early slowdown for consistently approaching small objects with predicted temporal overlap near stopping distance; preserve measured approach velocity before classification. Release lost tentative off-path candidates after 0.6s only when no entry is predicted within 0.8s. Confirmed crossings and near-path occlusions retain protection. Record actual Leo perception, route, control, signal, speed-cap and pose topics.

Validation: 175 function tests + 57 avoidance unit tests + 65 follower tests (297). Recorded 19:55:43 snapshots: 16.42s preserves crossing #15 zero cap and removes #17; 18.49s and 24.53s remove #17 zero cap. Snapshot checks do not establish closed-loop collision avoidance. Python syntax and reverse patch checks passed.

Rollback request:
이번 S5 미확정 보행자 접근 조기 감속·경로 밖 소실 후보 정지 해제·실제 제어 토픽 기록 추가만 롤백해줘. 기존 CP4~R1 903번 R1 전용 주행, S4 회피, S5 앞차 추종·옆 차량 제외와 확정된 횡단 보행자 보호는 유지하고 `rollback/2026-10-29_s5_pedestrian_candidate_lifecycle_before/changes.patch`만 역적용해줘.
