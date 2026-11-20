import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import physiologyTrains as t

class TrainTests(unittest.TestCase):
    def setUp(self):
        self.settings=t.TrainSettings()
        self.frames=pd.DataFrame(dict(frame=range(300),seconds=[f/15 for f in range(300)],roi_id=['left_vessel_group']*300,accepted=[True]*300))
        self.pulses=pd.DataFrame([dict(video='test',camera='left',roi_id='left_vessel_group',vessel_id='left_column_1',pulse_id=f'p{i}',source_event_id=f'e{i}',
            start_s=1+i*1.4,end_s=5+i*1.4,duration_s=4,x_px=900,y_px=800+i*15,fiber_fraction=0,boundary_uncertain=False) for i in range(3)])
    def test_overlapping_downward_pulses_form_a_train(self):
        result=t.search_trains(self.pulses,self.frames,15,self.settings)
        self.assertEqual(len(result),1);self.assertEqual(result.iloc[0].pulse_count,3)
        self.assertAlmostEqual(result.iloc[0].net_down_px,30)
    def test_complete_train_review_is_not_truncated_at_twenty_seconds(self):
        import tempfile,cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(1000,640))
            for _ in range(160):writer.write(np.zeros((640,1000,3),np.uint8))
            writer.release()
            pulses=self.pulses.copy();pulses['start_frame']=[10,50,100];pulses['end_frame']=[40,80,130]
            pulses['y_px']=[610,620,630]
            trains=pd.DataFrame([dict(video='source.avi',train_id='t1',pulse_ids='p0;p1;p2',vessel_id='left_column_1',camera='left',start_s=2,end_s=26,duration_s=24,pulse_count=3,flags='test')])
            out=Path(tmp)/'review';t.write_train_reviews(path,pulses,trains,out,5)
            cap=cv2.VideoCapture(str(out/'t1.mp4'));count=int(cap.get(7));cap.release()
            self.assertGreater(count/5,26)

    def test_two_pulses_are_insufficient(self):
        self.assertTrue(t.search_trains(self.pulses.iloc[:2],self.frames,15,self.settings).empty)
    def test_upward_train_and_same_spot_do_not_pass(self):
        for ys in [[830,815,800],[800,800,800]]:
            p=self.pulses.copy();p['y_px']=ys
            self.assertTrue(t.search_trains(p,self.frames,15,self.settings).empty)
    def test_cylinders_cannot_be_linked(self):
        p=self.pulses.copy();p.loc[2,'vessel_id']='left_column_2'
        self.assertTrue(t.search_trains(p,self.frames,15,self.settings).empty)
    def test_rejected_frames_break_train_evidence(self):
        self.frames.loc[30,'accepted']=False
        self.assertTrue(t.search_trains(self.pulses,self.frames,15,self.settings).empty)
    def test_flickering_same_source_is_not_three_pulses(self):
        p=self.pulses.copy();p['source_event_id']='same'
        self.assertTrue(t.search_trains(p,self.frames,15,self.settings).empty)
    def test_hot_pixel_lifetime_cannot_inflate_pulse_duration(self):
        rows=[]
        for f in range(90):
            rows.append(dict(event_id='e',video='v',camera='left',roi_id='left_vessel_group',frame=f,
                area_px=20 if f==40 else 5,excess_intensity=400 if f==40 else 5,
                x_px=900,y_px=800,core_overlap=0,flags=''))
        self.assertTrue(t.extract_pulses(pd.DataFrame(rows),self.frames,15,self.settings).empty)

if __name__=='__main__':unittest.main()
