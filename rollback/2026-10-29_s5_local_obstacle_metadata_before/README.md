# Actual Leo obstacle metadata for S5 vehicle exclusion

The real lidar_obstacles_local producer now publishes measured length/width/yaw, confirmed observation evidence, and stable saved-lane labels. Short predicted tracks preserve shape and revalidate lane membership; only S5 reliable stable predictions preserve motion classification. Existing radius, motion estimation and planner collision/crossing logic remain unchanged.

Validation: 91 tests passed, including actual producer-to-planner parallel-vehicle exclusion, same-lane following, short dropout, small-object protection, and existing follower safety tests. Python parse and reverse patch checks passed. A new full simulator drive has not been run.

Rollback request:
이번 S5 실제 인지 노드의 차량 형상·차선 정보 전달 및 짧은 추적 소실 정보 유지 변경만 롤백해줘. 기존 S4 S자 회피·출구 연결과 S5 트레일링·돌발 보행자·강아지 대응은 유지하고 `rollback/2026-10-29_s5_local_obstacle_metadata_before/changes.patch`만 역적용해줘.
