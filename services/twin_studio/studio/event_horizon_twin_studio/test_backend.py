import base64, json, math, subprocess, threading, unittest, urllib.request, urllib.error
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from lab_server import LabHandler
from simulation import simulate, validate, normalize_routes
C={'duration':80,'scale':.1,'seed':42,'randomCount':2,'routes':[{'id':'r','name':'North path','color':'#ff633e','points':[{'x':200,'y':260},{'x':600,'y':260}],'count':40,'speed':1.3,'width':3}], 'events':[{'id':'s','name':'Named surge','type':'surge','routeId':'r','at':.2,'start':10,'end':20,'people':10},{'id':'c','name':'Named collapse','type':'collapse','routeId':'r','at':.6,'start':25,'end':45,'people':0}]}
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),partial(LabHandler,directory=str(Path(__file__).parent/'dist')))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def post(self,path,data):
        req=urllib.request.Request(self.url+path,json.dumps(data).encode(),{'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(req) as r:return json.load(r)
    def test_health_and_static(self):
        with urllib.request.urlopen(self.url+'/api/health') as r:self.assertEqual(json.load(r)['engine'],'python')
        for file in ['index.html','app.js','engine.js','twin3d.js','venue.js','bootstrap.js','style.css']:
            with urllib.request.urlopen(self.url+'/'+file) as r:self.assertEqual(r.status,200)
    def test_picture_upload(self):
        png='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO0sAAAAASUVORK5CYII='
        self.assertTrue(self.post('/api/venue-image',{'image':'data:image/png;base64,'+png})['ok'])
    def test_route_import(self):
        old={'schema_version':'path-editor-v1','coordinate_system':'assumed_local_metres','routes':[{'route_id':'original','name':'My route','points':[[20,26],[60,26]]}]}
        got=self.post('/api/routes',{'payload':old,'scale':.1})['routes'][0]
        self.assertEqual(got['points'],C['routes'][0]['points']);self.assertEqual(got['id'],'original')
    def test_three_runs_and_csv_download(self):
        d=self.post('/api/render',C);self.assertEqual(set(d['results']),{'ideal','custom','random'})
        self.assertEqual([d['results'][m]['total'] for m in ['ideal','custom','random']],[40,50,40])
        self.assertEqual(d['results']['ideal']['events'],[])
        for name in ['01_ideal_flow.csv','02_custom_disasters.csv','03_random_collapses.csv','run_manifest.json']:
            with urllib.request.urlopen(self.url+'/api/runs/'+d['runId']+'/'+name) as r:
                self.assertIn('attachment',r.headers['Content-Disposition']);self.assertTrue(r.read())
    def test_bad_time_rejected_at_api(self):
        bad=json.loads(json.dumps(C));bad['events'][0]['start']=30
        with self.assertRaises(urllib.error.HTTPError) as cm:self.post('/api/render',bad)
        self.assertEqual(cm.exception.code,400)
    def test_invalid_image_rejected(self):
        with self.assertRaises(urllib.error.HTTPError):self.post('/api/venue-image',{'image':'data:image/png;base64,'+base64.b64encode(b'not an image').decode()})
    def test_cross_origin_rejected(self):
        req=urllib.request.Request(self.url+'/api/render',json.dumps(C).encode(),{'Content-Type':'application/json','Origin':'http://other.example'},method='POST')
        with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code,403)
    def test_js_python_parity(self):
        script="const E=require('./dist/engine.js');let c=JSON.parse(require('fs').readFileSync(0,'utf8'));console.log(JSON.stringify(Object.fromEntries(['ideal','custom','random'].map(m=>[m,E.simulate(c,m)]))))"
        js=json.loads(subprocess.check_output(['node','-e',script],input=json.dumps(C).encode()))
        def compare(a,b):
            if isinstance(a,dict):
                self.assertEqual(set(a),set(b))
                for k in a:compare(a[k],b[k])
            elif isinstance(a,list):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):compare(x,y)
            elif isinstance(a,(float,int)) and not isinstance(a,bool):self.assertAlmostEqual(a,b,places=8)
            else:self.assertEqual(a,b)
        for mode in js:compare(simulate(C,mode),js[mode])
if __name__=='__main__':unittest.main(verbosity=2)
