#!/usr/bin/env python3
"""Vessel-ROI pulse discovery. Candidates are not biological classifications."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VERSION = 1


@dataclass(frozen=True)
class Settings:
    seed_threshold: int = 40
    boundary_threshold: int = 20
    min_area: int = 2
    preferred_min_area: int = 10
    preferred_max_area: int = 2000
    background_threshold: int = 20
    max_lit_fraction: float = 0.15
    max_background_median: float = 8
    saturation_threshold: int = 250
    link_radius_px: float = 12
    max_missing_frames: int = 1
    site_radius_px: float = 15
    calibration_step: int = 15
    fiber_amplitude: float = 60
    fiber_duty: float = 0.02
    fiber_active_excess: float = 20
    trace_radius_px: int = 20
    review_padding_s: float = 2
    review_count: int = 8
    ambiguous_count: int = 2
    review_min_duration_s: float = 1.0
    preferred_duration_s: float = 2.0
    review_include_tiny: bool = False

    def __post_init__(self):
        if not 0 < self.review_min_duration_s <= self.preferred_duration_s:
            raise ValueError('Require 0 < review minimum <= preferred duration')
        if not 0 < self.boundary_threshold <= self.seed_threshold <= 255:
            raise ValueError('Require 0 < boundary_threshold <= seed_threshold <= 255')
        if not 0 <= self.max_lit_fraction <= 1 or not 0 <= self.fiber_duty <= 1:
            raise ValueError('Fractions must lie between zero and one')
        if not 1 <= self.min_area <= self.preferred_min_area <= self.preferred_max_area:
            raise ValueError('Invalid component size preferences')
        if min(self.link_radius_px,self.site_radius_px,self.trace_radius_px,self.calibration_step) <= 0:
            raise ValueError('Radii and calibration step must be positive')
        if min(self.max_missing_frames,self.review_count,self.ambiguous_count,self.review_padding_s) < 0:
            raise ValueError('Gap and review limits cannot be negative')


# Broad, provisional vessel-group regions from the local reference projection.
# These are NOT asserted individual-vessel boundaries. Replace with polygons
# drawn on a bright frame; coordinates are full side-by-side video pixels.
DEFAULT_REGIONS = [
    dict(id="left_vessel_group", camera="left", polygon=[[780, 640], [1430, 640], [1430, 1080], [780, 1080]]),
    dict(id="right_vessel_group", camera="right", polygon=[[2660, 620], [3150, 620], [3150, 1080], [2660, 1080]]),
]


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid frame rate: {path}")
    return cap, fps, (int(cap.get(3)), int(cap.get(4)))


def compile_regions(regions, size):
    width, height = size
    if not regions or len({r['id'] for r in regions}) != len(regions):
        raise ValueError("Supply nonempty, uniquely named regions")
    result = []
    occupied = np.zeros((height, width), np.uint8)
    for region in regions:
        points = np.asarray(region['polygon'], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("A region requires at least three polygon vertices")
        if (points < 0).any() or (points[:, 0] > width).any() or (points[:, 1] > height).any():
            raise ValueError(f"Region {region['id']} exceeds video dimensions; redraw the regions")
        x0, y0 = points.min(axis=0)
        x1, y1 = np.minimum(points.max(axis=0) + 1, [width, height])
        mask = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillPoly(mask, [points - [x0, y0]], 1)
        if not mask.any():
            raise ValueError("Empty region")
        if (occupied[y0:y1, x0:x1] & mask).any():
            raise ValueError("Regions must not overlap (would double-count detections)")
        occupied[y0:y1, x0:x1] |= mask
        result.append(dict(**region, x=int(x0), y=int(y0), w=int(x1-x0), h=int(y1-y0), mask=mask.astype(bool)))
    return result


def crop_gray(frame, region):
    x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
    return cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)


def frame_quality(gray, mask, settings):
    pixels = gray[mask]
    median = float(np.median(pixels))
    fraction = float(np.mean(pixels > settings.background_threshold))
    rejected = median > settings.max_background_median or fraction > settings.max_lit_fraction
    return dict(background_median=median, lit_fraction=fraction,
                saturated_fraction=float(np.mean(pixels >= settings.saturation_threshold)),
                accepted=not rejected, rejection_reason='broad_illumination' if rejected else '')


def calibrate(path, regions, settings, output_dir):
    """Sample local appearance; recurrent cores are provisional fiber flags only."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cap, fps, size = open_video(path)
    compiled = compile_regions(regions, size)
    stacks = {r['id']: [] for r in compiled}
    projection = np.zeros((size[1], size[0], 3), np.uint8)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for f in range(0, count, settings.calibration_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Calibration decode failed at frame {f}")
            projection = np.maximum(projection, frame)
            for r in compiled:
                gray = crop_gray(frame, r)
                if frame_quality(gray, r['mask'], settings)['accepted']:
                    stacks[r['id']].append(gray)
    finally:
        cap.release()
    model = dict(size=size, regions=regions, reference=str(Path(path).resolve()), fps=fps, sources={})
    sizes = []
    for r in compiled:
        frames = stacks[r['id']]
        if len(frames) < 10:
            raise RuntimeError(f"Only {len(frames)} dark calibration samples for {r['id']}; use a different reference")
        stack = np.stack(frames)
        baseline = np.median(stack, axis=0).astype(np.float32)
        high = np.percentile(stack, 99, axis=0)
        duty = np.mean(stack > baseline + settings.fiber_amplitude, axis=0)
        core = (high - baseline >= settings.fiber_amplitude) & (duty >= settings.fiber_duty) & r['mask']
        model['sources'][r['id']] = dict(baseline=baseline, core=core)
        for gray in frames:
            for d in detect_blobs(gray, r, model['sources'][r['id']], settings):
                sizes.append(dict(roi_id=r['id'], area_px=d['area_px'], width_px=d['width_px'], height_px=d['height_px']))
        cv2.polylines(projection, [np.asarray(r['polygon'], np.int32)], True, (0, 255, 0), 3)
        cv2.putText(projection, r['id'], (r['x'], max(25, r['y']-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite(str(out/'regions_reference.jpg'), projection)
    pd.DataFrame(sizes, columns=['roi_id','area_px','width_px','height_px']).to_csv(out/'calibration_components.csv', index=False)
    arrays = {f'{rid}__{k}': v for rid, source in model['sources'].items() for k,v in source.items()}
    np.savez_compressed(out/'calibration.npz', **arrays)
    metadata = {k:v for k,v in model.items() if k != 'sources'}
    metadata.update(settings=asdict(settings), version=VERSION,
                    warning='Candidate-light sizes and recurrent cores are not confirmed biological sizes or known fiber labels.')
    (out/'calibration.json').write_text(json.dumps(metadata, indent=2))
    return model


def detect_blobs(gray, region, source, settings):
    mask = region['mask']
    binary = ((gray >= settings.boundary_threshold) & mask).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    baseline = source['baseline']
    for label in range(1, n):
        x,y,w,h,area = map(int, stats[label])
        if area < settings.min_area:
            continue
        local = labels[y:y+h, x:x+w] == label
        values = gray[y:y+h,x:x+w][local]
        if values.max() < settings.seed_threshold:
            continue
        excess = values.astype(float) - baseline[y:y+h,x:x+w][local]
        core_overlap = float(source['core'][y:y+h,x:x+w][local].mean())
        # Local annulus excludes all segmented light and pixels outside the ROI.
        ax,ay,bx,by = max(0,x-5),max(0,y-5),min(gray.shape[1],x+w+5),min(gray.shape[0],y+h+5)
        background = gray[ay:by,ax:bx][(labels[ay:by,ax:bx] == 0) & mask[ay:by,ax:bx]]
        bg = float(np.median(background)) if len(background) else float(np.median(gray[mask]))
        contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        flags = []
        if area < settings.preferred_min_area: flags.append('tiny')
        if area > settings.preferred_max_area: flags.append('large')
        if core_overlap > 0: flags.append('recurrent_core_overlap')
        if (values >= settings.saturation_threshold).any(): flags.append('saturated')
        if x == 0 or y == 0 or x+w == gray.shape[1] or y+h == gray.shape[0]: flags.append('crop_edge')
        rows.append(dict(x_px=float(centroids[label,0]+region['x']), y_px=float(centroids[label,1]+region['y']),
                         area_px=area, bbox_x_px=x+region['x'], bbox_y_px=y+region['y'], width_px=w,height_px=h,
                         peak_intensity=int(values.max()),mean_intensity=float(values.mean()),
                         local_background=bg, integrated_intensity=float(values.astype(float).sum()-bg*area),
                         excess_intensity=float(np.maximum(excess,0).sum()),
                         core_overlap=core_overlap, saturated_fraction=float((values >= settings.saturation_threshold).mean()),
                         perimeter_px=perimeter, flags=';'.join(flags)))
    return rows


DETECTION_COLUMNS = ['video','roi_id','camera','frame','seconds','time_utc','time_panama','night_date','detection_id',
                     'x_px','y_px','area_px','bbox_x_px','bbox_y_px','width_px','height_px','peak_intensity','mean_intensity',
                     'local_background','integrated_intensity','excess_intensity','core_overlap','saturated_fraction','perimeter_px','flags']
EVENT_COLUMNS = ['video','roi_id','camera','event_id','start_frame','end_frame','peak_frame','start_s','end_s','duration_s',
                 'detected_frames','missing_frames','x_px','y_px','max_displacement_px','peak_area_px','peak_intensity',
                 'peak_excess','max_core_overlap','fiber_coincident_fraction','left_censored','right_censored','score','flags','site_id']


def link_events(detections, frames, fps, settings):
    """Fixed-anchor association: missing rejected frames always split events."""
    events, assignments = [], {}
    for roi_id, fq in frames.groupby('roi_id', sort=False):
        subset = detections[detections.roi_id == roi_id]
        by_frame = {int(f):g.to_dict('records') for f,g in subset.groupby('frame')}
        active, completed = [], []
        accepted = dict(zip(fq.frame.astype(int), fq.accepted.astype(bool)))
        for frame in fq.frame.astype(int):
            remaining=[]
            for track in active:
                if not accepted[frame] or frame-track[-1]['frame'] > settings.max_missing_frames+1:
                    completed.append(track)
                else: remaining.append(track)
            active = remaining
            if not accepted[frame]: continue
            candidates = by_frame.get(frame, [])
            distances = []
            for ti,t in enumerate(active):
                # Anchor stays fixed at first detection; no cumulative random walk.
                for di,d in enumerate(candidates):
                    distance = math.hypot(d['x_px']-t[0]['x_px'],d['y_px']-t[0]['y_px'])
                    if distance <= settings.link_radius_px: distances.append((distance,ti,di))
            used_t, used_d = set(), set()
            for _,ti,di in sorted(distances):
                if ti not in used_t and di not in used_d:
                    active[ti].append(candidates[di]);used_t.add(ti);used_d.add(di)
            active.extend([d] for i,d in enumerate(candidates) if i not in used_d)
        completed.extend(active)
        for track in completed:
            first,last = track[0],track[-1]
            event_id = f'e{len(events)+1:06d}'
            for d in track: assignments[d['detection_id']] = event_id
            peak = max(track,key=lambda d:d['excess_intensity'])
            cx,cy = np.median([(d['x_px'],d['y_px']) for d in track],axis=0)
            displacement = max(math.hypot(d['x_px']-cx,d['y_px']-cy) for d in track)
            overlap = max(d['core_overlap'] for d in track)
            left = not accepted.get(first['frame']-1,False)
            right = not accepted.get(last['frame']+1,False)
            flags=set(filter(None,';'.join(d['flags'] for d in track).split(';')))
            coincidence=sum('fiber_active' in d['flags'] for d in track)/len(track)
            if len(track)==1: flags.add('single_frame')
            if left or right: flags.add('boundary_uncertain')
            span=last['frame']-first['frame']+1
            if span/fps < settings.review_min_duration_s: flags.add('short_for_review')
            score=math.log1p(peak['excess_intensity'])*min(len(track)/(fps*settings.preferred_duration_s),1)/(1+displacement/5)*(1-.8*overlap)
            score *= 1-.3*coincidence
            if peak['area_px'] < settings.preferred_min_area: score*=.25
            if peak['area_px'] > settings.preferred_max_area: score*=.25
            if len(track)>fps*10: score*=.25;flags.add('persistent')
            events.append(dict(video=first['video'],roi_id=roi_id,camera=first['camera'],event_id=event_id,
                start_frame=first['frame'],end_frame=last['frame'],peak_frame=peak['frame'],start_s=first['seconds'],end_s=last['seconds'],
                duration_s=span/fps,detected_frames=len(track),missing_frames=span-len(track),x_px=cx,y_px=cy,
                max_displacement_px=displacement,peak_area_px=peak['area_px'],peak_intensity=peak['peak_intensity'],
                peak_excess=peak['excess_intensity'],max_core_overlap=overlap,fiber_coincident_fraction=coincidence,left_censored=left,right_censored=right,
                score=score,flags=';'.join(sorted(flags)),site_id=''))
    result=pd.DataFrame(events,columns=EVENT_COLUMNS)
    sites=[]
    for i,e in result.iterrows():
        eligible=[(math.hypot(e.x_px-s['x_px'],e.y_px-s['y_px']),s) for s in sites if s['roi_id']==e.roi_id]
        eligible=[(d,s) for d,s in eligible if d<=settings.site_radius_px]
        if eligible:site=min(eligible,key=lambda item:item[0])[1]
        else:
            site=dict(site_id=f's{len(sites)+1:05d}',roi_id=e.roi_id,camera=e.camera,x_px=e.x_px,y_px=e.y_px);sites.append(site)
        result.at[i,'site_id']=site['site_id']
    return result.sort_values('score',ascending=False), assignments, pd.DataFrame(sites,columns=['site_id','roi_id','camera','x_px','y_px'])


def scan(path, start_utc, model, settings, output_dir, write_traces=True, write_reviews_output=True):
    """Write completed results last; settings/model fingerprints prevent stale reuse."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'complete.json').unlink(missing_ok=True)
    cap,fps,size=open_video(path)
    if tuple(model['size']) != size:
        cap.release();raise ValueError('Video dimensions differ from calibration; redraw regions')
    regions=compile_regions(model['regions'],size)
    if start_utc is None:
        cap.release();raise ValueError('Missing recording start time')
    start=pd.Timestamp(start_utc)
    if start.tzinfo is None:
        cap.release();raise ValueError('start_utc must be timezone aware')
    records,qualities=[],[];frame_number=0
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            time_utc=start.tz_convert('UTC')+pd.Timedelta(seconds=frame_number/fps)
            local=time_utc.tz_convert('America/Panama')
            common=dict(video=Path(path).name,frame=frame_number,seconds=frame_number/fps,
                        time_utc=time_utc.isoformat(),time_panama=local.isoformat(),night_date=(local-pd.Timedelta(hours=12)).date().isoformat())
            for r in regions:
                gray=crop_gray(frame,r);source=model['sources'][r['id']]
                q=frame_quality(gray,r['mask'],settings)
                blobs=detect_blobs(gray,r,source,settings) if q['accepted'] else []
                core_values=gray[source['core']]
                context=dict(**common,roi_id=r['id'],camera=r['camera'])
                qualities.append(dict(**context,**q,detection_count=len(blobs),
                                      fiber_core_mean=float(core_values.mean()) if len(core_values) else None))
                # A localized fiber must not be diluted by other inactive cores.
                core_excess = core_values.astype(float) - source['baseline'][source['core']]
                fiber_active = bool(len(core_excess) and core_excess.max() > settings.fiber_active_excess)
                for d in blobs:
                    if fiber_active: d['flags'] = ';'.join(filter(None,[d['flags'],'fiber_active']))
                    records.append(dict(**context,detection_id=f'd{len(records)+1:08d}',**d))
            frame_number+=1
    finally:cap.release()
    if frame_number==0 or (expected>0 and frame_number<expected):
        raise RuntimeError(f'Incomplete decode: {frame_number}/{expected} frames; no completion marker written')
    detections=pd.DataFrame(records,columns=DETECTION_COLUMNS);frames=pd.DataFrame(qualities)
    events,assignments,sites=link_events(detections,frames,fps,settings)
    detections['event_id']=detections.detection_id.map(assignments)
    events['start_time_utc'] = events.start_s.map(lambda seconds: (start + pd.Timedelta(seconds=seconds)).tz_convert('UTC').isoformat())
    events['end_time_utc'] = events.end_s.map(lambda seconds: (start + pd.Timedelta(seconds=seconds)).tz_convert('UTC').isoformat())
    events['start_time_panama'] = events.start_s.map(lambda seconds: (start + pd.Timedelta(seconds=seconds)).tz_convert('America/Panama').isoformat())
    for name,table in [('frames',frames),('detections',detections),('candidates',events),('sites',sites)]:
        table.to_csv(out/f'{name}.csv',index=False)
    if write_traces:
        write_site_traces(path,regions,model,settings,frames,sites,out)
    reviews=write_reviews(path,regions,settings,events,fps,out) if write_reviews_output else 0
    info=dict(version=VERSION,video=Path(path).name,fps=fps,decoded_frames=frame_number,settings=asdict(settings),
              regions=model['regions'],reference=model['reference'],fingerprint=fingerprint(model,settings),
              start_utc=start.isoformat(),video_bytes=Path(path).stat().st_size,
              candidates=len(events),reviews=reviews,site_traces_written=write_traces,
              note='Scores rank candidates, not probabilities. Duration is observed frame span; missing frames and censored boundaries remain explicit. Sites are local to each video.')
    (out/'complete.json').write_text(json.dumps(info,indent=2))
    return events


def fingerprint(model,settings):
    h=hashlib.sha256(json.dumps(dict(version=VERSION,settings=asdict(settings),regions=model['regions'],size=model['size']),sort_keys=True).encode())
    h.update(Path(__file__).read_bytes())
    for rid,source in sorted(model['sources'].items()):
        h.update(rid.encode())
        for key,value in sorted(source.items()):h.update(key.encode());h.update(value.tobytes())
    return h.hexdigest()


def write_site_traces(path,regions,model,settings,frames,sites,out):
    import csv
    columns=['video','roi_id','camera','site_id','frame','seconds','time_utc','time_panama','accepted','aperture_pixels',
             'mean_intensity','peak_intensity','excess_intensity','saturated_fraction']
    prepared={}
    for r in regions:
        points=[]
        for s in sites[sites.roi_id==r['id']].to_dict('records'):
            cx,cy=s['x_px']-r['x'],s['y_px']-r['y']
            radius=settings.trace_radius_px
            x0,y0=max(0,math.floor(cx-radius)),max(0,math.floor(cy-radius))
            x1,y1=min(r['w'],math.ceil(cx+radius)+1),min(r['h'],math.ceil(cy+radius)+1)
            yy,xx=np.indices((y1-y0,x1-x0))
            aperture=((xx+x0-cx)**2+(yy+y0-cy)**2<=radius**2)&r['mask'][y0:y1,x0:x1]
            baseline=model['sources'][r['id']]['baseline'][y0:y1,x0:x1][aperture]
            points.append((s,x0,y0,x1,y1,aperture,baseline,int(aperture.sum())))
        prepared[r['id']]=points
    lookup={(row.roi_id,int(row.frame)):row for row in frames.itertuples()}
    cap,_,_=open_video(path);frame_number=0
    try:
        with (out/'site_traces.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
            while True:
                ok,frame=cap.read()
                if not ok:break
                for r in regions:
                    gray=crop_gray(frame,r);q=lookup[(r['id'],frame_number)]
                    for s,x0,y0,x1,y1,aperture,baseline,aperture_pixels in prepared[r['id']]:
                        row=dict(video=Path(path).name,roi_id=r['id'],camera=r['camera'],site_id=s['site_id'],frame=frame_number,
                                 seconds=q.seconds,time_utc=q.time_utc,time_panama=q.time_panama,accepted=q.accepted,aperture_pixels=aperture_pixels)
                        if q.accepted:
                            v=gray[y0:y1,x0:x1][aperture]
                            row.update(mean_intensity=float(v.mean()),peak_intensity=int(v.max()),
                                       excess_intensity=float(np.maximum(v.astype(float)-baseline,0).sum()),
                                       saturated_fraction=float((v>=settings.saturation_threshold).mean()))
                        writer.writerow(row)
                frame_number+=1
    finally:cap.release()
    if frame_number != frames.frame.nunique():raise RuntimeError('Incomplete site-trace decode')


def select_reviews(events, settings, fps):
    # Avoid using the entire shortlist on several reflections in one blink.
    events=events[events.duration_s >= settings.review_min_duration_s]
    if not settings.review_include_tiny:
        events=events[events.peak_area_px >= settings.preferred_min_area]
    chosen=[]
    for index,e in events.iterrows():
        if len(chosen) >= settings.review_count: break
        if any(e.roi_id == events.loc[i,'roi_id'] and abs(e.peak_frame-events.loc[i,'peak_frame']) < fps for i in chosen): continue
        chosen.append(index)
    selected=events.loc[chosen]
    rest=events.drop(index=chosen)
    if len(rest) and settings.ambiguous_count:
        idx=np.unique(np.linspace(0,len(rest)-1,min(settings.ambiguous_count,len(rest)),dtype=int))
        selected=pd.concat([selected,rest.iloc[idx]])
    return selected


def write_reviews(path,regions,settings,events,fps,out):
    import html
    reviews=out/'review';reviews.mkdir(exist_ok=True)
    selected=select_reviews(events,settings,fps)
    selected.to_csv(reviews/'selection.csv',index=False)
    cards=[]
    if selected.empty:
        cards.append(f'<p><strong>No candidates meet the current review criteria.</strong> Minimum duration: {settings.review_min_duration_s:g} s; tiny components included: {settings.review_include_tiny}. All detections remain in the CSV files. This does not establish that the video contains no bioluminescence.</p>')
    region_map={r['id']:r for r in regions}
    cap,_,_=open_video(path);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for e in selected.itertuples():
            r=region_map[e.roi_id];pad=round(settings.review_padding_s*fps)
            first=max(0,e.start_frame-pad);last=min(total-1,e.end_frame+pad)
            # At most 20 seconds per review; CSV retains full event bounds.
            if last-first+1>round(20*fps):
                first=max(first,e.peak_frame-round(10*fps));last=min(last,first+round(20*fps)-1)
            scale=min(2,640/r['w']);w=int(r['w']*scale)//2*2;h=int(r['h']*scale)//2*2
            clip=reviews/f'{e.event_id}.mp4'
            writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
            if not writer.isOpened():raise RuntimeError('Cannot create review video')
            snapshots=[]
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES,first)
                for f in range(first,last+1):
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError('Review decode failed')
                    crop=frame[r['y']:r['y']+r['h'],r['x']:r['x']+r['w']].copy()
                    cv2.circle(crop,(round(e.x_px-r['x']),round(e.y_px-r['y'])),settings.trace_radius_px,(0,255,0),1)
                    crop=cv2.resize(crop,(w,h))
                    cv2.putText(crop,f'{e.event_id} {f/fps:.2f}s', (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                    writer.write(crop)
                    if f in {first,e.peak_frame,last}:snapshots.append(crop)
            finally:writer.release()
            # Browser-compatible video when ffmpeg is installed; retain playable
            # MPEG-4 otherwise without adding a required Python dependency.
            import shutil, subprocess
            if shutil.which('ffmpeg'):
                encoded=reviews/f'{e.event_id}_h264.mp4'
                command=['ffmpeg','-v','error','-y','-i',str(clip),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)]
                conversion=subprocess.run(command,capture_output=True)
                if conversion.returncode == 0: encoded.replace(clip)
                else: encoded.unlink(missing_ok=True)
            sheet=reviews/f'{e.event_id}.jpg'
            cv2.imwrite(str(sheet),np.hstack(snapshots))
            cards.append(f'<article><h2>{html.escape(e.roi_id)} / {e.event_id} — {e.start_s:.2f}–{e.end_s:.2f}s; score {e.score:.2f}</h2><p>{html.escape(e.flags)}</p><a href="{clip.name}">Open cropped MP4</a><br><img width="100%" src="{sheet.name}"></article>')
    finally:cap.release()
    (reviews/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pulse candidates</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#171717;color:#eee}a{color:#8cf}article{margin:35px 0}</style><h1>Candidate light events</h1><p>Unconfirmed candidates. Green circle is a fixed analysis aperture, not a segmented contour. MP4 files can be opened in a local video player. Long events show up to 20 seconds around the peak.</p>'+''.join(cards))
    return len(selected)


def load_calibration(directory):
    directory=Path(directory)
    info=json.loads((directory/'calibration.json').read_text())
    with np.load(directory/'calibration.npz', allow_pickle=False) as archive:
        info['sources']={r['id']:{key:archive[f"{r['id']}__{key}"].copy() for key in ('baseline','core')} for r in info['regions']}
    return info


def is_complete(directory, model, settings, video_name, start_utc):
    directory=Path(directory)
    required=['complete.json','frames.csv','detections.csv','candidates.csv','sites.csv','site_traces.csv','review/index.html','review/selection.csv']
    if not all((directory/name).exists() for name in required): return False
    try:
        info=json.loads((directory/'complete.json').read_text())
        return (info['fingerprint']==fingerprint(model,settings) and info['video']==video_name
                and pd.Timestamp(info['start_utc'])==pd.Timestamp(start_utc))
    except (KeyError,ValueError): return False
