# S4 continuous tracking and exit

Changes: remove binary 5m prefix stop; limit speed by actual blocker stopping distance only. Extend completed return paths with validated saved-route continuation. S4 follower accepts short forward targets and retains saved-route target when no local forward sample remains, retaining planner caps. Provisional static memories expire after 0.6s without observation; confirmed stable multi-scan shapes retain existing occlusion protection. Other sectors unchanged.

Validation: 169 function tests, 57 avoidance unit tests, 85 follower tests passed. Successful-run exit snapshot previously stopped with 4.57m remaining now returns LOCAL_PATH, 60m checked continuation, 20km/h cap. This is a snapshot replay, not a closed-loop simulator validation. Reverse patch check passed.

Rollback request:
이번 S4 5m 미만 강제 정지 제거·짧은 경로 추종·출구 정상 경로 연결·미확정 장애물 기억 만료 변경만 롤백해줘. 기존 S4 S자 회피·FTG·디버그 표시와 S5 트레일링·옆 차선 차량 제외·돌발 장애물 대응은 유지하고 `rollback/2026-10-29_s4_continuous_exit_before/changes.patch`만 역적용해줘.
