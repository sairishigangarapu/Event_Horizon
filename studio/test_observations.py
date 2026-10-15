import copy,json,tempfile,threading,unittest,urllib.request,urllib.error
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from lab_server import LabHandler
from simulation import simulate
from observations import generate_observations,available_at,export_separated,sources_for
from test_backend import C
class ObservationTests(unittest.TestCase):
    def setUp(self):self.c=copy.deepcopy(C);self.result=simulate(self.c,'custom')
    def test_deterministic_schema(self):
        rows=generate_observations(self.c,self.result)
        self.assertEqual(rows,generate_observations(self.c,self.result))
        for r in rows:self.assertEqual(set(r),{'observation_id','source_id','source_kind','route_id','observed_at_s','available_at_s','values'})
        for key in ['seed','scheduled','blocked','completed','incident_labels','s_m']:
            self.assertNotIn('"'+key+'"',json.dumps(rows))
    def test_delay_boundary(self):
        rows=generate_observations(self.c,self.result);self.assertEqual(available_at(rows,0),[])
        row=next(r for r in rows if r['source_kind']=='camera')
        self.assertNotIn(row,available_at(rows,row['available_at_s']-.01));self.assertIn(row,available_at(rows,row['available_at_s']))
        for t in [0,5,20,45,80]:self.assertTrue(all(r['available_at_s']<=t and r['observed_at_s']<=t for r in available_at(rows,t)))
    def test_future_independence(self):
        changed=copy.deepcopy(self.c);changed['events'].append({'id':'future','name':'Future surge','type':'surge','routeId':'r','at':.5,'start':60,'end':70,'people':20})
        self.assertEqual(available_at(generate_observations(self.c,self.result),30),available_at(generate_observations(changed,simulate(changed,'custom')),30))
    def test_coverage_and_missing(self):
        for change in [{'route_ids':[]},{'missing_probability':1}]:
            self.c['observationSources']=[{**sources_for(C)[0],**change}];self.assertEqual(generate_observations(self.c,self.result),[])
    def test_bad_sources(self):
        for key,val in [('delay_s',-1),('interval_s',0),('relative_noise',float('nan')),('route_ids',['unknown'])]:
            self.c['observationSources']=[{**sources_for(C)[0],key:val}]
            with self.assertRaises(ValueError):sources_for(self.c)
    def test_bad_cutoffs(self):
        for t in [-1,float('nan'),float('inf'),'10',True]:
            with self.assertRaises(ValueError):available_at([],t)
    def test_export_and_inference_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run='a'*32;folder=root/run;folder.mkdir();export_separated(folder,self.c,{m:simulate(self.c,m) for m in ['ideal','custom','random']})
            self.assertTrue((folder/'hidden_ground_truth/custom/incident_labels.jsonl').is_file())
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(LabHandler,directory=str(folder)));server.inference_only=True;server.observation_root=root
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(url+'/api/observations/'+run+'/custom?time_s=20') as r:data=json.load(r)
                self.assertTrue(all(r['available_at_s']<=20 for r in data['observations']))
                for path in ['/', '/hidden_ground_truth/scenario.json','/observable/custom.jsonl','/api/runs/'+run+'/run_manifest.json','/../hidden_ground_truth/scenario.json']:
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+path)
                    self.assertEqual(error.exception.code,404)
                for query in ['', '?time_s=NaN','?time_s=inf','?time_s=-1','?time_s=1&truth=1']:
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+'/api/observations/'+run+'/custom'+query)
                    self.assertEqual(error.exception.code,400)
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(urllib.request.Request(url+'/api/render',b'{}',{'Content-Type':'application/json'}))
                self.assertEqual(error.exception.code,403)
            finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main(verbosity=2)
