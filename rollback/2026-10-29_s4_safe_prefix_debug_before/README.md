# S4 safe-prefix tracking and debug display

Only this change is captured in changes.patch; earlier S4 FTG and S5 fixes are preserved.

Validation: 166 function tests, 53 avoidance unit tests, 82 follower tests, 22 map debug tests passed. Debug source/stale/exit checks passed. Latest stopped snapshot now retains LOCAL_PATH with a positive safe-prefix speed cap; full closed-loop passage has not been verified.

Rollback request:
이번 S4 안전한 전방 경로 추종·FTG 선택 보완·디버그 상태 표시 변경만 롤백해줘. 기존 S4 회피 및 FTG 기능과 S5 옆 차선 차량 대응 수정은 유지하고 `rollback/2026-10-29_s4_safe_prefix_debug_before/changes.patch`만 역적용해줘.
