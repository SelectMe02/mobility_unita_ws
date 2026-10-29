# S4 validated path priority

The S4 prefix validator emits explicit body_validated evidence for the selected path. The node transports it as s4_body_validated. Only in S4, with static_pass_managed, a fresh selected route and this boolean, the follower uses the selected path/cap instead of the second circle-based pass veto. Route/traffic zero caps and stale input stops remain binding. The legacy S4 pass check no longer subtracts the front overhang twice. Other sectors retain their checks.

Validation: 111 integrated follower/fallback tests, 169 route function tests, 42 avoidance unit tests passed (322 total). Includes redundant-circle veto removal, zero/traffic caps, stale input, S5 scope isolation and S4 distance correction. Python syntax and reverse patch checks passed. No full simulator run performed.

Rollback request:
이번 S4 검증된 로컬·FTG 경로 우선 적용 및 앞범퍼 중복 차감 수정만 롤백해줘. 기존 S4 S자 회피·출구 연결·기억 관리와 S5 옆 차량 대응·트레일링·돌발 장애물 대응은 유지하고 `rollback/2026-10-29_s4_validated_path_priority_before/changes.patch`만 역적용해줘.
