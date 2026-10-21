"""Recorded S5 obstacle-driven pedal pulses; no control publication."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False
base=1792337220.
def rows(n):return [r for r in json.loads((P/n).read_text()) if 1792337273<=r['t']<=1792337275.8]
r=rows('leo_route_avoidance.json');c=rows('leo_raw_ctrl_cmd.json');v=rows('competition_ego_speed_kmh.json')
f,a=plt.subplots(3,1,figsize=(11,7.8),sharex=True)
a[0].step([x['t']-base for x in r],[min(30,x['data']['speed_cap_kmh']) if x['data']['speed_cap_kmh'] is not None else 30 for x in r],where='post',color='#b91c1c',lw=2,label='회피 계획 속도 상한 (제약 없으면 30 표시)')
a[0].set_ylabel('속도 상한 (km/h)');a[0].set_ylim(-1,35)
a[1].step([x['t']-base for x in c],[x['data']['brake'] for x in c],where='post',color='#b91c1c',label='신호 제어 전 브레이크',lw=2)
a[1].step([x['t']-base for x in c],[x['data']['accel'] for x in c],where='post',color='#15803d',label='신호 제어 전 액셀',lw=1.8)
a[1].set_ylabel('페달 명령');a[1].set_ylim(-.03,1.1)
a[2].plot([x['t']-base for x in v],[x['data'] for x in v],color='#1e3a8a',lw=2,label='실제 속도')
a[2].set_ylabel('차량 속도 (km/h)');a[2].set_xlabel('2026-10-19 00:27:xx (한국 시각, 초)')
for ax in a:
 ax.axvspan(53.355,53.472,color='#f97316',alpha=.15)
 ax.axvspan(55.212,55.330,color='#ef4444',alpha=.2)
 ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9);ax.set_xlim(53,55.8)
a[0].annotate('한 프레임 정적 후보\n17.37 → 해제',xy=(53.4,17.37),xytext=(53.75,7),arrowprops={'arrowstyle':'->'},fontsize=10)
a[0].annotate('군집 합침·분리 뒤 횡단 판단\n0 → 해제',xy=(55.24,0),xytext=(54.1,5),arrowprops={'arrowstyle':'->'},fontsize=10)
f.suptitle('S5: 차선 변경 없이 장애물 판단 변화로 가속·제동 반복',fontsize=15,fontweight='bold')
f.text(.015,.01,'기록된 경로 옆 이동량 0m. 원본: semi_leo_2026-10-19-00-25-29.bag. 신호 제어기가 개입하기 전 페달 명령을 표시.',fontsize=9)
f.tight_layout(rect=[0,.035,1,.96]);f.savefig(P/'static_obstacle_chatter.png',dpi=160)
print(P/'static_obstacle_chatter.png')
