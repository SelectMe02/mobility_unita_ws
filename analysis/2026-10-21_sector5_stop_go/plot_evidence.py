"""Plot recorded S5 control evidence; reads files only, publishes no ROS messages."""
import json,csv,math,datetime,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
plt.rcParams['font.family']='Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus']=False
base=1792337220.0
start,end=1792337270.,1792337280.
def load(name):return json.loads((P/name).read_text())
def selected(name):return [r for r in load(name) if start<=r['t']<=end]
signal=selected('leo_signal_status.json')
route=selected('leo_route_avoidance.json')
speed=selected('competition_ego_speed_kmh.json')
raw=selected('leo_raw_ctrl_cmd.json')
fig,axs=plt.subplots(4,1,figsize=(12.8,11.3),sharex=True,gridspec_kw={'height_ratios':[1.4,1,1,1.2]})
x=lambda rows:[r['t']-base for r in rows]
axs[0].plot(x(speed),[r['data'] for r in speed],color='#172554',lw=2,label='실제 속도')
axs[0].step(x(route),[r['data']['speed_cap_kmh'] if r['data']['speed_cap_kmh'] is not None else np.nan for r in route],where='post',color='#dc2626',lw=1.6,label='장애물 회피 속도 상한')
axs[0].step(x(signal),[r['data']['speed_cap_kmh'] if r['data'].get('speed_cap_kmh') is not None else np.nan for r in signal],where='post',color='#a16207',lw=1.7,label='신호 제어 속도 상한')
axs[0].set_ylim(-1,38);axs[0].set_ylabel('속도 (km/h)')
axs[1].step(x(signal),[r['data']['command']['brake'] for r in signal],where='post',lw=2.2,color='#dc2626',label='최종 브레이크')
axs[1].step(x(signal),[r['data']['input_command']['brake'] for r in signal],where='post',lw=1.6,linestyle='--',color='#1d4ed8',label='신호 제어 전 브레이크')
axs[1].set_ylabel('브레이크 명령');axs[1].set_ylim(-.06,1.12)
axs[2].step(x(signal),[r['data']['command']['accel'] for r in signal],where='post',lw=1.8,color='#15803d',label='최종 액셀')
axs[2].set_ylabel('액셀 명령');axs[2].set_ylim(-.06,1.12)
for ident,color,label in [(100,'#dc2626','횡단 물체 #100'),(107,'#64748b','고정 물체 #107'),(99,'#94a3b8','고정 물체 #99')]:
 obs=[(r['t']-base,o) for r in route for o in r['data']['obstacles'] if o['id']==ident]
 axs[3].plot([a for a,o in obs],[o['x'] for a,o in obs],color=color,lw=2,label=label)
axs[3].axhline(-60.1453,color='#172554',ls=':',label='정지 중 자차 X')
axs[3].set_ylabel('지도 X 좌표 (m)');axs[3].set_xlabel('2026-10-19 00:27:xx (한국 시각, 초)')
for ax in axs:
 ax.axvspan(52.588,52.985,color='#eab308',alpha=.18)
 ax.axvspan(55.211,55.330,color='#f97316',alpha=.25)
 ax.axvspan(56.791,57.808,color='#ef4444',alpha=.12)
 ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9,ncol=3)
 ax.set_xlim(50,60)
axs[0].text(52.65,35,'신호 미검출',fontsize=10,color='#854d0e')
axs[0].text(56.85,30,'횡단 물체 → 정지',fontsize=10,color='#991b1b')
axs[3].text(57.12,-64.5,'자차가 정지한 동안에도\n물체는 옆으로 이동',fontsize=10,color='#991b1b')
fig.suptitle('Sector 5 끝: 신호 미검출 급제동 후, 횡단 물체에 대한 정지',fontsize=16,fontweight='bold')
fig.text(.01,.006,'출처: semi_leo_2026-10-19-00-25-29.bag. 음영: 신호 unknown / 짧은 횡단 예측 / 장애물 속도 상한 0. 촬영 영상은 기록되지 않음.',fontsize=9,color='#475569')
fig.tight_layout(rect=[0,.025,1,.97]);fig.savefig(P/'sector5_stop_go.png',dpi=170)
with (P/'signal_control_timeline.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['time_kst','speed_kmh','signal_state','signal','signal_cap_kmh','input_accel','input_brake','output_accel','output_brake'])
 for r in signal:
  d=r['data'];tim=datetime.datetime.fromtimestamp(r['t'],datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='milliseconds')
  w.writerow([tim,d.get('speed_kmh'),d.get('state'),d.get('signal'),d.get('speed_cap_kmh'),d['input_command']['accel'],d['input_command']['brake'],d['command']['accel'],d['command']['brake']])
for name in ['s5_raw_cluster_evidence.json','leo_signal_analysis.py']:
 src=Path('/tmp')/name
 if src.exists():shutil.copy2(src,P/name)
print(P/'sector5_stop_go.png')
