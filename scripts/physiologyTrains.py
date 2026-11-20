"""Search exported detections for overlapping, downward pulse trains.

Timing references: Hensley et al. 2023, Table 1, PMC10685132.
Published means are ranking guides, not biological exclusion limits.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainSettings:
    min_pulses: int = 3
    max_pulses: int = 12
    min_pulse_s: float = .7
    max_pulse_s: float = 15
    max_detection_gap_s: float = .2
    min_signal_area: int = 6
    min_peak_area: int = 10
    min_excess_per_pixel: float = 5
    min_signal_coverage: float = .6
    min_onset_gap_s: float = .3
    max_onset_gap_s: float = 6
    max_train_s: float = 40
    lateral_tolerance_px: float = 35
    upward_tolerance_px: float = 5
    min_separation_px: float = 5
    min_net_down_px: float = 8
    beam_width: int = 12

    def __post_init__(self):
        if not 3 <= self.min_pulses <= self.max_pulses:
            raise ValueError('Train search requires at least three pulses')
        if not 0 < self.min_pulse_s <= self.max_pulse_s <= self.max_train_s:
            raise ValueError('Invalid pulse/train duration limits')
        if not 0 < self.min_onset_gap_s <= self.max_onset_gap_s:
            raise ValueError('Invalid onset interval limits')
        if not 0 < self.min_signal_coverage <= 1:
            raise ValueError('Coverage must be in (0, 1]')
        if min(self.min_signal_area,self.min_peak_area,self.min_excess_per_pixel,self.lateral_tolerance_px,self.min_separation_px,self.min_net_down_px,self.beam_width) <= 0:
            raise ValueError('Size, signal, spatial, and beam limits must be positive')
        if min(self.upward_tolerance_px,self.max_detection_gap_s) < 0:
            raise ValueError('Gap/reversal tolerances cannot be negative')


# Provisional columns enclosing individual vessels, to prevent linking across
# neighboring cylinders. Boundaries require user review; no centimetre scale.
COLUMNS = {'left': [820, 960, 1090, 1240, 1410, 1570],
           'right': [2500, 2660, 2790, 2940, 3110, 3270]}
PULSE_COLUMNS = ['video','camera','roi_id','vessel_id','pulse_id','source_event_id','start_frame','end_frame',
                 'start_s','end_s','duration_s','signal_frames','coverage','x_px','y_px','peak_area_px',
                 'mean_excess_per_pixel','core_overlap','fiber_fraction','boundary_uncertain']
TRAIN_COLUMNS = ['video','camera','roi_id','vessel_id','train_id','pulse_ids','pulse_count','start_s','end_s',
                 'duration_s','net_down_px','lateral_span_px','mean_onset_gap_s','fiber_fraction','score','flags']


def extract_pulses(detections, frames, fps, settings):
    """Measure sustained brightening, not the lifetime of an underlying hot pixel."""
    rows=[]
    rejected={roi: g.loc[~g.accepted,'frame'].to_numpy() for roi,g in frames.groupby('roi_id')}
    last_frame=int(frames.frame.max())
    for event_id, group in detections.groupby('event_id',sort=False):
        group=group.sort_values('frame')
        signal=group[(group.area_px >= settings.min_signal_area) &
                     (group.excess_intensity/group.area_px >= settings.min_excess_per_pixel)]
        if signal.empty: continue
        records=signal.to_dict('records');segments=[];current=[]
        for row in records:
            if current:
                gap=row['frame']-current[-1]['frame']
                bad=rejected[row['roi_id']]
                barrier=np.any((bad>current[-1]['frame']) & (bad<row['frame']))
                if gap>max(1,round(settings.max_detection_gap_s*fps)+1) or barrier:
                    segments.append(current);current=[]
            current.append(row)
        segments.append(current)
        for segment in segments:
            first,last=segment[0],segment[-1]
            span=last['frame']-first['frame']+1;duration=span/fps;coverage=len(segment)/span
            if not settings.min_pulse_s<=duration<=settings.max_pulse_s or coverage<settings.min_signal_coverage:continue
            if max(r['area_px'] for r in segment)<settings.min_peak_area:continue
            x,y=np.median([(r['x_px'],r['y_px']) for r in segment],axis=0)
            edges=COLUMNS[first['camera']];column=np.searchsorted(edges,x,side='right')-1
            if not 0<=column<len(edges)-1:continue
            # Do not silently assign an event that wanders into another cylinder.
            if any(not edges[column]<=r['x_px']<edges[column+1] for r in segment):continue
            bad=rejected[first['roi_id']]
            uncertain=first['frame']==0 or last['frame']==last_frame or first['frame']-1 in bad or last['frame']+1 in bad
            rows.append(dict(video=first['video'],camera=first['camera'],roi_id=first['roi_id'],
                vessel_id=f"{first['camera']}_column_{column+1}",pulse_id=f'p{len(rows)+1:06d}',source_event_id=event_id,
                start_frame=first['frame'],end_frame=last['frame'],start_s=first['frame']/fps,end_s=last['frame']/fps,
                duration_s=duration,signal_frames=len(segment),coverage=coverage,x_px=x,y_px=y,
                peak_area_px=max(r['area_px'] for r in segment),
                mean_excess_per_pixel=float(np.mean([r['excess_intensity']/r['area_px'] for r in segment])),
                core_overlap=max(r['core_overlap'] for r in segment),
                fiber_fraction=np.mean(['fiber_active' in str(r['flags']) for r in segment]),boundary_uncertain=bool(uncertain)))
    return pd.DataFrame(rows,columns=PULSE_COLUMNS)


def search_trains(pulses, frames, fps, settings):
    found=[]
    for vessel,table in pulses.groupby('vessel_id',sort=False):
        records=table.sort_values(['start_s','pulse_id']).to_dict('records');paths=[]
        for i,pulse in enumerate(records):
            options=[([i],0.)]
            for j,previous in enumerate(records[:i]):
                dt=pulse['start_s']-previous['start_s']
                dx=abs(pulse['x_px']-previous['x_px']);dy=pulse['y_px']-previous['y_px']
                if not settings.min_onset_gap_s<=dt<=settings.max_onset_gap_s:continue
                if dx>settings.lateral_tolerance_px or dy < -settings.upward_tolerance_px:continue
                if math.hypot(dx,dy)<settings.min_separation_px:continue
                for chain,score in paths[j]:
                    if len(chain)>=settings.max_pulses:continue
                    # A source fragmented by flicker must not count as multiple locations.
                    if any(records[k]['source_event_id']==pulse['source_event_id'] or
                           math.hypot(records[k]['x_px']-pulse['x_px'],records[k]['y_px']-pulse['y_px'])<settings.min_separation_px for k in chain):continue
                    end=max([records[k]['end_s'] for k in chain]+[pulse['end_s']])+1/fps
                    if end-records[chain[0]]['start_s']>settings.max_train_s:continue
                    gap_score=math.exp(-abs(math.log(dt/1.4)))
                    options.append((chain+[i],score+1+gap_score-dx/(2*settings.lateral_tolerance_px)))
            paths.append(sorted(options,key=lambda item:(len(item[0]),item[1]),reverse=True)[:settings.beam_width])
            for chain,score in paths[-1]:
                if len(chain)<settings.min_pulses:continue
                seq=[records[k] for k in chain];ys=np.array([p['y_px'] for p in seq])
                if ys[-1]-ys[0]<settings.min_net_down_px or np.mean(np.diff(ys)>0)<.6:continue
                first=seq[0];end=max(p['end_s'] for p in seq)+1/fps
                q=frames[(frames.roi_id==first['roi_id']) & (frames.seconds>=first['start_s']) & (frames.seconds<end)]
                if not q.accepted.all():continue
                fiber=float(np.mean([p['fiber_fraction'] for p in seq]))
                duration_score=np.mean([math.exp(-abs(math.log(p['duration_s']/(5.711 if k==0 else 3.5)))) for k,p in enumerate(seq)])
                score=(score+duration_score)*(1-.5*fiber)
                found.append(dict(video=first['video'],camera=first['camera'],roi_id=first['roi_id'],vessel_id=vessel,train_id='',
                    pulse_ids=';'.join(p['pulse_id'] for p in seq),pulse_count=len(seq),start_s=first['start_s'],end_s=end,
                    duration_s=end-first['start_s'],net_down_px=ys[-1]-ys[0],lateral_span_px=np.ptp([p['x_px'] for p in seq]),
                    mean_onset_gap_s=float(np.mean(np.diff([p['start_s'] for p in seq]))),fiber_fraction=fiber,score=score,
                    flags='provisional_vessel_columns'+(';fiber_coincident' if fiber else '')+(';boundary_uncertain' if any(p['boundary_uncertain'] for p in seq) else '')))
    retained=[]
    for row in sorted(found,key=lambda r:r['score'],reverse=True):
        ids=set(row['pulse_ids'].split(';'))
        if any(len(ids & set(r['pulse_ids'].split(';')))/min(len(ids),r['pulse_count'])>=.67 for r in retained):continue
        row['train_id']=f't{len(retained)+1:05d}';retained.append(row)
    return pd.DataFrame(retained,columns=TRAIN_COLUMNS)


def analyze_directory(directory,settings=TrainSettings(),output_name="trains"):
    directory=Path(directory);info=json.loads((directory/'complete.json').read_text())
    detections=pd.read_csv(directory/'detections.csv').fillna({'flags':''})
    frames=pd.read_csv(directory/'frames.csv')
    pulses=extract_pulses(detections,frames,info['fps'],settings)
    trains=search_trains(pulses,frames,info['fps'],settings)
    out=directory/output_name;out.mkdir(exist_ok=True)
    pulses.to_csv(out/'pulses.csv',index=False);trains.to_csv(out/'trains.csv',index=False)
    (out/'settings.json').write_text(json.dumps(dict(settings=asdict(settings),columns=COLUMNS,
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC10685132/',
        note='Heuristic pixel/timing tolerances; no physical scale. Overlap is allowed. No joins across videos. Raw detection thresholds still limit sensitivity.'),indent=2))
    return pulses,trains


def write_train_reviews(video_path, pulses, trains, destination, fps, max_reviews=12):
    """Render the full train plus context; never apply the old 20-second cap."""
    import cv2
    import html
    import shutil
    import subprocess
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    selected=trains.head(max_reviews)
    selected.to_csv(out/'selection.csv',index=False)
    cards=[]
    if selected.empty:
        cards.append('<p><strong>No qualifying downward trains were found.</strong> This is a detector result, not proof that bioluminescence is absent. The search is limited by the original blob thresholds, provisional vessel columns, and recording boundaries.</p>')
    else:
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {video_path}')
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));height=int(cap.get(4))
        try:
            for e in selected.itertuples():
                seq=pulses.set_index('pulse_id').loc[e.pulse_ids.split(';')].reset_index()
                column=int(e.vessel_id.rsplit('_',1)[1])-1;edges=COLUMNS[e.camera]
                x0,x1=edges[column],edges[column+1];y0=min(600,height-2);y1=height
                width=(x1-x0)*2//2*2;h=(y1-y0)*2//2*2
                first=max(0,int(math.floor(e.start_s*fps))-round(3*fps))
                last=min(total-1,int(math.ceil(e.end_s*fps))+round(3*fps))
                clip=out/f'{e.train_id}.mp4';writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,h))
                if not writer.isOpened():raise RuntimeError('Cannot write train review')
                peak_frames={int(round((p.start_frame+p.end_frame)/2)) for p in seq.itertuples()};snapshots=[]
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,first)
                    for frame_index in range(first,last+1):
                        ok,frame=cap.read()
                        if not ok:raise RuntimeError('Incomplete train review decode')
                        crop=frame[y0:y1,x0:x1].copy()
                        for number,p in enumerate(seq.itertuples(),1):
                            active=p.start_frame<=frame_index<=p.end_frame
                            color=(0,255,0) if active else (100,100,100)
                            point=(round(p.x_px-x0),round(p.y_px-y0))
                            cv2.circle(crop,point,12,color,1);cv2.putText(crop,str(number),(point[0]+12,point[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                        crop=cv2.resize(crop,(width,h))
                        cv2.putText(crop,f'{frame_index/fps:.2f}s',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
                        writer.write(crop)
                        if frame_index in peak_frames:snapshots.append(crop)
                finally:writer.release()
                if shutil.which('ffmpeg'):
                    encoded=out/f'{e.train_id}_h264.mp4'
                    conversion=subprocess.run(['ffmpeg','-v','error','-y','-i',str(clip),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],capture_output=True)
                    if conversion.returncode==0:encoded.replace(clip)
                    else:encoded.unlink(missing_ok=True)
                if snapshots:cv2.imwrite(str(out/f'{e.train_id}.jpg'),np.hstack(snapshots))
                cards.append(f'<article><h2>{html.escape(e.video)} / {e.train_id}: {e.pulse_count} pulses, {e.duration_s:.2f}s</h2><p>{html.escape(e.vessel_id)}; {html.escape(e.flags)}</p><video controls preload="metadata" src="{e.train_id}.mp4"></video><p><a href="{e.train_id}.mp4">Open full train clip</a></p></article>')
        finally:cap.release()
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Downward pulse trains</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto;background:#17191d;color:#eee}a{color:#8cf}article{margin:30px 0}video{max-width:100%;max-height:700px}</style><h1>Candidate downward pulse trains</h1><p>At least three distinct locations with successive onsets and overall downward progression. Green numbers indicate active pulses; grey numbers show other pulse positions. Overlap is allowed. These are unconfirmed candidates.</p>'+''.join(cards))
    return len(selected)


def write_night_summary(night_directory, modes=('trains','trains_exploratory')):
    """Refresh the summary from CSVs after every parameter change."""
    import html
    root=Path(night_directory);rows=[];links=[];seconds=0
    for marker in sorted(root.glob('*/complete.json')):
        info=json.loads(marker.read_text());seconds+=info['decoded_frames']/info['fps']
        for mode in modes:
            folder=marker.parent/mode
            if not (folder/'trains.csv').exists():continue
            pulses=pd.read_csv(folder/'pulses.csv');trains=pd.read_csv(folder/'trains.csv')
            rows.append(dict(video=marker.parent.name,mode=mode,pulse_candidates=len(pulses),train_candidates=len(trains)))
            prefix=f'{marker.parent.name}/{mode}'
            links.append(f'<p>{html.escape(marker.parent.name)} / {html.escape(mode)}: <a href="{prefix}/pulses.csv">possible pulses</a> · <a href="{prefix}/trains.csv">trains</a> · <a href="{prefix}/settings.json">settings</a></p>')
    summary=pd.DataFrame(rows,columns=['video','mode','pulse_candidates','train_candidates'])
    summary.to_csv(root/'train_search_summary.csv',index=False)
    count=int(summary.train_candidates.sum()) if len(summary) else 0
    outcome='No qualifying downward trains found.' if not count else f'{count} candidate train records across search modes (may overlap). These are not confirmed bioluminescence.'
    image='<h2>Check provisional vessel columns</h2><img width="100%" src="vessel_columns.jpg">' if (root/'vessel_columns.jpg').exists() else ''
    (root/'trains_index.html').write_text('<!doctype html><meta charset="utf-8"><title>Downward train search</title><style>body{font:16px system-ui;background:#17191d;color:#eee;max-width:1100px;margin:30px auto;padding:20px}a{color:#8cf}td,th{padding:8px;text-align:left}</style>'
        +f'<h1>{html.escape(root.name)}: downward pulse-train search</h1><p><strong>{outcome}</strong></p><p>{summary.video.nunique()} videos; {seconds/60:.2f} minutes of footage. This samples only part of the observation night.</p>'
        +'<p>The standard and exploratory passes require at least three distinct pulse locations with successive onsets and overall downward progression. Overlap is allowed. Timing and pixel tolerances are adjustable heuristics for small cylinders, not confidence intervals from the paper. Provisional vessel columns prevent linking neighboring cylinders.</p><p>The exploratory pass includes tiny possible artifacts. Signals missed or merged by the original blob detector cannot be recovered here. Excluded frames and video boundaries break evidence. A zero result does not prove absence of bioluminescence.</p><p><a href="train_search_summary.csv">Summary CSV</a> · <a href="index.html">Earlier individual-event gallery</a></p>'
        +summary.to_html(index=False)+image+'<h2>Detailed exports</h2>'+''.join(links)
        +'<p>For any newly found trains, render complete review clips from the source video with <code>write_train_reviews</code>. Previously generated individual-event clips are not complete-train evidence.</p>')
    return summary
