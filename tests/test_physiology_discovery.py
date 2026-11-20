import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('discovery',Path(__file__).resolve().parents[1]/'scripts/physiologyDiscovery.py')
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.settings=d.Settings(review_count=1,ambiguous_count=0,review_min_duration_s=.1)
        self.regions=[dict(id='vessel',camera='left',polygon=[[0,0],[95,0],[95,63],[0,63]])]
        self.region=d.compile_regions(self.regions,(96,64))[0]
        self.source=dict(baseline=np.zeros((64,96),np.float32),core=np.zeros((64,96),bool))

    def test_stationary_core_is_flagged_not_removed(self):
        gray=np.zeros((64,96),np.uint8);gray[20:25,30:35]=150
        self.source['core'][20:25,30:35]=True
        blobs=d.detect_blobs(gray,self.region,self.source,self.settings)
        self.assertEqual(len(blobs),1)
        self.assertEqual(blobs[0]['area_px'],25)
        self.assertEqual(blobs[0]['core_overlap'],1)

    def test_black_field_cannot_hide_bright_vessel(self):
        gray=np.full((64,96),100,np.uint8)
        self.assertFalse(d.frame_quality(gray,self.region['mask'],self.settings)['accepted'])
        gray[:]=0;gray[20:25,30:35]=255
        self.assertTrue(d.frame_quality(gray,self.region['mask'],self.settings)['accepted'])

    def test_fixed_anchor_prevents_drifting_track(self):
        rows=[]
        for frame,x in enumerate([10,16,22,28,34]):
            rows.append(dict(video='v',roi_id='vessel',camera='left',frame=frame,seconds=frame/15,
                detection_id=str(frame),x_px=x,y_px=20,area_px=25,peak_intensity=150,excess_intensity=100,
                core_overlap=0,flags=''))
        frames=pd.DataFrame(dict(roi_id=['vessel']*5,frame=range(5),accepted=[True]*5))
        events,_,_=d.link_events(pd.DataFrame(rows),frames,15,self.settings)
        self.assertEqual(len(events),2)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            d.compile_regions(self.regions+[dict(self.regions[0],id='duplicate')],(96,64))

    def test_end_to_end_dark_gaps_rejection_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sbs_20260828_235910.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(96,64))
            self.assertTrue(writer.isOpened())
            for f in range(30):
                frame=np.zeros((64,96,3),np.uint8)
                if 4<=f<=8 or 10<=f<=14:frame[20:25,30:35]=180
                if f==9:frame[:]=180
                writer.write(frame)
            writer.release()
            model=dict(size=(96,64),regions=self.regions,reference=str(path),sources={'vessel':self.source})
            out=Path(tmp)/'out'
            events=d.scan(path,pd.Timestamp('2026-08-28T23:59:10Z'),model,self.settings,out)
            self.assertEqual(len(events),2)
            self.assertTrue(events.right_censored.any());self.assertTrue(events.left_censored.any())
            self.assertTrue(np.allclose(events.duration_s,5/15))
            self.assertEqual(events.site_id.nunique(),1)
            frames=pd.read_csv(out/'frames.csv')
            self.assertEqual(len(frames),30)
            self.assertFalse(frames.loc[9,'accepted'])
            traces=pd.read_csv(out/'site_traces.csv')
            self.assertTrue(np.isnan(traces.loc[9,'mean_intensity']))
            self.assertEqual(traces.loc[0,'peak_intensity'],0)
            self.assertTrue((out/'complete.json').exists())
            self.assertTrue((out/'review/index.html').exists())
            cap=cv2.VideoCapture(str(next((out/'review').glob('*.mp4'))))
            self.assertTrue(cap.read()[0]);cap.release()

    def test_duration_shortlist_does_not_backfill_with_flashes_or_tiny_spots(self):
        events=pd.DataFrame([
            dict(event_id='flash',duration_s=.2,peak_area_px=50,roi_id='vessel',peak_frame=1),
            dict(event_id='tiny',duration_s=3,peak_area_px=5,roi_id='vessel',peak_frame=60),
            dict(event_id='pulse',duration_s=3,peak_area_px=50,roi_id='vessel',peak_frame=120),
        ])
        selected=d.select_reviews(events,d.Settings(),15)
        self.assertEqual(selected.event_id.tolist(),['pulse'])
        self.assertTrue(d.select_reviews(events.iloc[:2],d.Settings(),15).empty)

    def test_cropped_site_apertures_match_full_roi_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gradient.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(96,64))
            yy,xx=np.indices((64,96))
            frame=np.repeat(((xx+yy)%180).astype(np.uint8)[...,None],3,axis=2)
            writer.write(frame);writer.release()
            cap=cv2.VideoCapture(str(path));ok,decoded=cap.read();cap.release()
            self.assertTrue(ok)
            gray=cv2.cvtColor(decoded,cv2.COLOR_BGR2GRAY)
            frames=pd.DataFrame([dict(roi_id='vessel',frame=0,seconds=0,time_utc='2026-09-01T00:00:00Z',time_panama='2026-08-31T19:00:00-05:00',accepted=True)])
            sites=pd.DataFrame([dict(site_id='edge',roi_id='vessel',camera='left',x_px=1.25,y_px=2.5),dict(site_id='middle',roi_id='vessel',camera='left',x_px=40.1,y_px=30.7)])
            model=dict(sources={'vessel':self.source})
            d.write_site_traces(path,[self.region],model,self.settings,frames,sites,Path(tmp))
            result=pd.read_csv(Path(tmp)/'site_traces.csv')
            for i,site in sites.iterrows():
                mask=((xx-site.x_px)**2+(yy-site.y_px)**2 <= self.settings.trace_radius_px**2)
                values=gray[mask]
                self.assertEqual(result.iloc[i].aperture_pixels,int(mask.sum()))
                self.assertAlmostEqual(result.iloc[i].mean_intensity,float(values.mean()))
                self.assertEqual(result.iloc[i].excess_intensity,float(values.sum()))

    def test_settings_reject_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            d.Settings(seed_threshold=10,boundary_threshold=20)

    def test_fingerprint_changes_with_model_and_settings(self):
        model=dict(size=(96,64),regions=self.regions,sources={'vessel':self.source})
        original=d.fingerprint(model,self.settings)
        self.assertNotEqual(original,d.fingerprint(model,replace(self.settings,seed_threshold=50)))
        self.source['core'][0,0]=True
        self.assertNotEqual(original,d.fingerprint(model,self.settings))

    def test_no_detections_still_has_csv_headers(self):
        frames=pd.DataFrame(dict(roi_id=['vessel'],frame=[0],accepted=[True]))
        events,assignments,sites=d.link_events(pd.DataFrame(columns=d.DETECTION_COLUMNS),frames,15,self.settings)
        self.assertTrue(events.empty);self.assertIn('duration_s',events.columns)
        self.assertTrue(sites.empty);self.assertFalse(assignments)


if __name__=='__main__':unittest.main()
