# CP4 through saved R1 point 903: R1 only

Configured in semi.launch as r1_only_point_ids=708-903. CP4 is duplicated as saved points 708/709; the deduplicated route uses 708 (index 697). Last point 903 maps to index 887. Route planner suppresses offset passes and clears active passes within the interval; follower forces R1 and suppresses external offset steering. All obstacle and signal speed checks remain active. Pedestrian detection/response logic is unchanged.

Validation: 171 function tests, 57 avoidance unit tests, 63 follower tests passed (291). Additional actual follower restriction and exact CP4 boundary checks passed. Syntax/XML and reverse patch checks passed. No closed-loop simulator test performed.

Rollback request:
이번 CP4부터 R1 903번 포인트까지 R1 전용 주행·차선 변경 제한만 롤백해줘. 기존 S4 회피와 S5 앞차 추종·옆 차량 제외·보행자 대응은 유지하고 `rollback/2026-10-29_cp4_r1_903_before/changes.patch`만 역적용해줘.
