#!/usr/bin/env python3
"""Resumable full-night physiology scan. Run with the manta Python environment."""
import argparse
import concurrent.futures
from dataclasses import asdict
import gzip
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import sys

import cv2
import numpy as np
import pandas as pd
import physiologyDiscovery as discovery
import physiologyTrains as trains
import brightnessOverTime as brightness

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/physiology_full_nights_Bocas2026'
NIGHTS=['2026-08-31','2026-09-02']
BOX='https://ucsb.box.com/s/dwav1fq9nqys8l1uve1fqdhoez33y4z2'
REGIONS=[dict(id='left_vessel_group',camera='left',polygon=[[820,600],[1570,600],[1570,1080],[820,1080]]),
         dict(id='right_vessel_group',camera='right',polygon=[[2500,600],[3270,600],[3270,1080],[2500,1080]])]
SETTINGS=discovery.Settings(seed_threshold=30,boundary_threshold=15,review_count=0,ambiguous_count=0)
EXPLORATORY=trains.TrainSettings(min_signal_area=2,min_peak_area=2,min_excess_per_pixel=2,min_pulse_s=.5,
    min_signal_coverage=.5,max_detection_gap_s=.3,max_onset_gap_s=8,min_net_down_px=5,lateral_tolerance_px=45,upward_tolerance_px=8)


def curl(url,destination):
    subprocess.run(['curl','-fLsS','--connect-timeout','15','--max-time','180','--retry','3','-o',str(destination),url],check=True)


def page(number):
    folder=OUT/'inventory_pages';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{number}.html'
    if not path.exists():curl(f'{BOX}?page={number}',path)
    for match in re.finditer(r'Box\.(?:postStreamData|prefetchedData)\s*=\s*(\{.*?\});',path.read_text(),re.S):
        data=json.loads(match.group(1))
        if '/app-api/enduserapp/shared-folder' in data:return data
    raise RuntimeError(f'Missing Box metadata on page {number}')


def inventory():
    data=page(1);count=int(data['/app-api/enduserapp/shared-folder']['pageCount']);rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for number,data in enumerate(pool.map(page,range(1,count+1)),1):
            for item in data['/app-api/enduserapp/shared-folder']['items']:
                if item.get('type')=='file' and item['name'].lower().endswith('.mkv'):
                    rows.append(dict(id=str(item['id']),name=item['name'],size_bytes=int(item['itemSize'])))
            if number%10==0:print(f'Inventory {number}/{count} pages',flush=True)
    (OUT/'shared_name.txt').write_text(data['/app-api/enduserapp/shared-item']['sharedName'])
    table=pd.DataFrame(rows).drop_duplicates('id')
    table['start_utc']=pd.to_datetime(table.name.map(brightness.parse_utc_start),utc=True)
    if table.start_utc.isna().any():raise ValueError('Unparseable recording timestamp')
    table['start_panama']=table.start_utc.dt.tz_convert('America/Panama')
    table['night_date']=(table.start_panama-pd.Timedelta(hours=12)).dt.date.astype(str)
    table=table[table.night_date.isin(NIGHTS)].sort_values('start_utc').reset_index(drop=True)
    if table.name.duplicated().any():raise ValueError('Duplicate video filenames require disambiguation')
    table.to_csv(OUT/'inventory.csv',index=False)
    print(table.groupby('night_date').agg(videos=('id','size'),first=('start_panama','min'),last=('start_panama','max'),bytes=('size_bytes','sum')).to_string(),flush=True)
    return table


def download(item,path):
    from urllib.parse import urlencode
    if path.exists() and path.stat().st_size==int(item['size_bytes']):return
    shared=(OUT/'shared_name.txt').read_text()
    query=urlencode(dict(rm='box_download_shared_file',shared_name=shared,file_id='f_'+str(item['id'])))
    curl('https://ucsb.app.box.com/index.php?'+query,path)
    if path.stat().st_size!=int(item['size_bytes']):raise RuntimeError('Downloaded file size mismatch: '+item['name'])


def prepare(table):
    for night in NIGHTS:
        group=table[table.night_date==night]
        if group.empty:raise RuntimeError(f'No recordings found for {night}')
        calibration=OUT/'calibration'/night;calibration.mkdir(parents=True,exist_ok=True)
        for ordinal in [0,len(group)//2,len(group)-1]:
            item=group.iloc[ordinal].to_dict();path=calibration/item['name'];download(item,path)
            cap,_,size=discovery.open_video(path);projection=np.zeros((size[1],size[0],3),np.uint8)
            for f in range(0,int(cap.get(7)),90):
                cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read()
                if ok:projection=np.maximum(projection,frame)
            cap.release()
            cv2.imwrite(str(calibration/f'alignment_{ordinal}.jpg'),cv2.resize(projection,(1920,540)))
            if ordinal==0 and not (calibration/'calibration.json').exists():
                discovery.calibrate(path,REGIONS,SETTINGS,calibration)
        print('Prepared calibration and alignment samples:',night,flush=True)


def completed(item):
    folder=OUT/item['night_date']/Path(item['name']).stem
    marker=folder/'full_complete.json'
    if not marker.exists():return False
    try:
        info=json.loads(marker.read_text())
        model=discovery.load_calibration(OUT/'calibration'/item['night_date'])
        expected=discovery.fingerprint(model,SETTINGS)
        return info['fingerprint']==expected and all((folder/name).exists() for name in ['frames.csv','detections.csv.gz','trains/trains.csv','trains_exploratory/trains.csv'])
    except (KeyError,ValueError,OSError):return False


def compress(path):
    destination=path.with_suffix(path.suffix+'.gz')
    temporary=destination.with_suffix('.gz.partial')
    with path.open('rb') as source,gzip.open(temporary,'wb',compresslevel=1) as target:shutil.copyfileobj(source,target,1024*1024)
    temporary.replace(destination);path.unlink()


def worker(item):
    cv2.setNumThreads(1)
    folder=OUT/item['night_date']/Path(item['name']).stem;folder.mkdir(parents=True,exist_ok=True)
    try:
        if completed(item):return dict(video=item['name'],status='cached',night=item['night_date'])
        if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free; pausing this video')
        started=time.time()
        model=discovery.load_calibration(OUT/'calibration'/item['night_date'])
        with tempfile.TemporaryDirectory(prefix='physiology_full_') as temp:
            path=Path(temp)/item['name'];download(item,path)
            events=discovery.scan(path,pd.Timestamp(item['start_utc']),model,SETTINGS,folder,
                                   write_traces=False,write_reviews_output=False)
            counts={}
            for mode,settings in [('trains',trains.TrainSettings()),('trains_exploratory',EXPLORATORY)]:
                pulse_table,train_table=trains.analyze_directory(folder,settings,mode)
                counts[mode]=len(train_table)
                if len(train_table):trains.write_train_reviews(path,pulse_table,train_table,folder/mode/'review',model['fps'],max_reviews=3)
            # Keep a small individual-pulse review set for investigation independent of the train criterion.
            eligible=discovery.select_reviews(events,discovery.Settings(seed_threshold=30,boundary_threshold=15,review_count=2,ambiguous_count=0),model['fps'])
            discovery.write_reviews(path,discovery.compile_regions(model['regions'],model['size']),
                discovery.Settings(seed_threshold=30,boundary_threshold=15,review_count=2,ambiguous_count=0),eligible,model['fps'],folder)
        compress(folder/'detections.csv')
        info=json.loads((folder/'complete.json').read_text())
        info.update(full_night=True,storage='Lossless gzip for detections.csv and site_traces.csv',train_counts=counts,elapsed_s=time.time()-started)
        (folder/'full_complete.json').write_text(json.dumps(info,indent=2))
        (folder/'failure.json').unlink(missing_ok=True)
        return dict(video=item['name'],night=item['night_date'],status='complete',seconds=round(time.time()-started,1),**counts)
    except Exception as error:
        (folder/'failure.json').write_text(json.dumps(dict(error=str(error),traceback=traceback.format_exc()),indent=2))
        return dict(video=item['name'],night=item['night_date'],status='failed',error=str(error))


def report(table):
    rows=[]
    for item in table.to_dict('records'):
        folder=OUT/item['night_date']/Path(item['name']).stem;marker=folder/'full_complete.json'
        row=dict(video=item['name'],night_date=item['night_date'],start_utc=item['start_utc'],start_panama=item['start_panama'],status='pending')
        if marker.exists():
            info=json.loads(marker.read_text());duration=info['decoded_frames']/info['fps']
            row.update(status='complete',frames=info['decoded_frames'],duration_s=duration,end_utc=(pd.Timestamp(item['start_utc'])+pd.Timedelta(seconds=duration)).isoformat(),
                trains=info['train_counts']['trains'],exploratory_trains=info['train_counts']['trains_exploratory'])
        elif (folder/'failure.json').exists():row.update(status='failed',error=json.loads((folder/'failure.json').read_text())['error'])
        rows.append(row)
    coverage=pd.DataFrame(rows);coverage.to_csv(OUT/'coverage.csv',index=False)
    summaries=[]
    for night,g in coverage.groupby('night_date'):
        good=g[g.status=='complete'];summaries.append(dict(night=night,total=len(g),completed=len(good),failed=int((g.status=='failed').sum()),hours=float(good.duration_s.sum()/3600) if len(good) else 0))
    status=pd.DataFrame(summaries)
    links=''.join(f'<li><a href="{n}/trains_index.html">{n}: train candidates</a></li>' for n in NIGHTS)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Full-night physiology scan</title><style>body{font:16px system-ui;margin:30px;background:#17191d;color:#eee}a{color:#8cf}td,th{padding:10px}</style><h1>Full-night physiology scan</h1><p>Panama observation nights of August 31 and September 2. This report refreshes every minute. Completed hours reflect decoded footage, not gaps between source recordings.</p>'+status.to_html(index=False)+'<p><a href="coverage.csv">Per-video coverage and failures CSV</a></p><ul>'+links+'</ul>')
    return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inventory','prepare','run','report']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='inventory':inventory();return
    table=pd.read_csv(OUT/'inventory.csv',dtype={'id':str})
    if args.stage=='prepare':prepare(table);return
    if args.stage=='report':print(report(table).to_string(index=False));return
    report(table)
    pending=[item for item in table.to_dict('records') if not completed(item)]
    print(f'Starting {len(pending)} pending recordings with {args.workers} workers',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,item) for item in pending]
        try:
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(future.result()),flush=True);report(table)
        except KeyboardInterrupt:
            for process in pool._processes.values():process.terminate()
            raise
    print(report(table).to_string(index=False),flush=True)
    for night in NIGHTS:trains.write_night_summary(OUT/night)

if __name__=='__main__':main()
