# S5 vehicle following

Scope: actual local follower suppresses new overtakes of vehicle-sized dynamic lead objects in S5. Existing adjacent-lane filter is retained. Route avoidance waits for stationary evidence before treating a newly observed vehicle-sized object as a static pass target in S5; measured motion and its short classification dropout history prevent static pass initiation. Obstacle data and crossing/braking checks are unchanged.

Validation: 169 function tests + 53 avoidance unit tests + 84 local follower tests passed. Includes same-lane following, release after adjacent-lane departure, motion label dropout, static vehicle eligibility and identical pedestrian/small-animal crossing outputs. Reverse patch check passed. Full simulator drive has not been run.

Rollback request:
이번 S5 동적 앞차 추월 방지·트레일링 및 차량 이동 판정 보완만 롤백해줘. 기존 S4 S자 회피·FTG·디버그 표시와 S5 옆 차선 차량 제외 및 돌발 보행자·강아지 대응은 유지하고 `rollback/2026-10-29_s5_vehicle_follow_before/changes.patch`만 역적용해줘.
