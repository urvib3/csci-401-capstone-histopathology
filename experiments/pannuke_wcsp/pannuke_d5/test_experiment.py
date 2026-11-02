"""Run with project Python: python -m unittest discover -s <experiment> -v."""
import json,unittest
from pathlib import Path
import numpy as np
from core import *
HERE=Path(__file__).resolve().parent

class ExperimentTests(unittest.TestCase):
    def test_probabilities(self):
        p=probabilities(np.zeros((164,164,6)));validate_probabilities(p)
        np.testing.assert_allclose(p,1/6)
        with self.assertRaises(AssertionError):validate_probabilities(np.ones((164,164,6)))
    def test_mapping(self):
        from run import mapping
        self.assertEqual(list(mapping()['verified_mapping'].values()),CLASSES)
    def test_crop_overlap(self):
        m=np.zeros((256,256,6));m[46,46,0]=1;m[209,209,4]=1;m[45,46,3]=1;m[50,50,0:2]=1
        y,n=semantic(m);self.assertEqual(y.shape,(164,164));self.assertEqual(y[0,0],1);self.assertEqual(y[-1,-1],5);self.assertEqual(y[4,4],2);self.assertEqual(n,1)
    def test_edge_counts(self):
        y=np.array([[0,1,2],[3,4,5],[1,2,3]],np.uint8)
        expected=np.zeros((6,6),int);edges=0
        for r in range(3):
            for s in range(3):
                for t in range(r*3+s+1,9):
                    rr,ss=divmod(t,3)
                    if abs(rr-r)<=2 and abs(ss-s)<=2:
                        expected[y[r,s],y[rr,ss]]+=1;expected[y[rr,ss],y[r,s]]+=1;edges+=1
        np.testing.assert_array_equal(adjacency(y),expected);self.assertEqual(int(adjacency(y).sum()),2*edges)
        same=adjacency(np.zeros((3,3),np.uint8));self.assertEqual(same[0,0],72)
    def test_tables(self):
        count=np.ones((6,6),int);np.fill_diagonal(count,500)
        f=fit(count)
        for k in ['raw_cost','data_cost','prior','final']:
            c=f[k];np.testing.assert_allclose(c,c.T);self.assertTrue((c>=0).all());np.testing.assert_array_equal(np.diag(c),0)
        self.assertAlmostEqual(f['final'][~np.eye(6,dtype=bool)].mean(),1)
        np.testing.assert_allclose(f['prior'][0],f['data_cost'][0])
    def test_beta_zero_and_energy(self):
        rng=np.random.default_rng(19);p=rng.random((8,9,6));p/=p.sum(-1,keepdims=True)
        c=np.ones((6,6))-np.eye(6);base=p.argmax(-1)
        y,info=solve(p,c,0);np.testing.assert_array_equal(y,base)
        y,info=solve(p,c,.30);self.assertTrue(all(np.diff(info['energy'])<=1e-7));self.assertTrue(info['verified_local_minimum'])
        self.assertEqual(improving_moves(y,unary(p),c,.30)[0],0)
        # Independent brute-force total-energy checks for every final label move.
        u=unary(p);e=energy(y,u,c,.30)
        for r in range(8):
            for s in range(9):
                for k in range(6):
                    z=y.copy();z[r,s]=k;self.assertGreaterEqual(energy(z,u,c,.30)+1e-8,e)
    def test_round_limit_status(self):
        p=np.random.default_rng(8).random((10,10,6));p/=p.sum(-1,keepdims=True)
        _,info=solve(p,np.ones((6,6))-np.eye(6),max_rounds=1)
        self.assertEqual(info['stop_reason'],'round_limit')
    def test_fixed_pooled_metrics(self):
        a=np.zeros((6,6),int);a[0,0]=90;a[1,1]=10
        m=metrics(a);self.assertAlmostEqual(m['biological_miou'],.2);self.assertAlmostEqual(m['six_class_miou'],2/6)
        self.assertEqual(m['gt_absent_classes'],[2,3,4,5])
        b=np.zeros((6,6),int);b[1,0]=30
        self.assertAlmostEqual(metrics(a+b)['iou'][1],.25)
        self.assertNotEqual(metrics(a+b)['biological_miou'],(metrics(a)['biological_miou']+metrics(b)['biological_miou'])/2)
    def test_bootstrap_reproducibility(self):
        a=np.array([np.eye(6,dtype=int),np.eye(6,dtype=int)*3]);b=a.copy();b[0,1,1]=0;b[0,1,0]=1
        self.assertEqual(bootstrap(a,b,replicates=100),bootstrap(a,b,replicates=100))
    def test_split(self):
        p=HERE/'split.json'
        if not p.exists():self.skipTest('prepare not yet run')
        x=json.loads(p.read_text());a={r['image_index'] for r in x['train']};b={r['image_index'] for r in x['test']}
        self.assertEqual(len(a),200);self.assertEqual(len(b),20);self.assertFalse(a&b)
        excluded=set(map(int,json.loads((HERE/'previous_use_exclusions.json').read_text())))
        self.assertFalse((a|b)&excluded)
        count=np.array([r['class_pixels'] for r in x['test']]);self.assertTrue((count[:,1:].sum(0)>0).all())
    def test_frozen_and_real_alignment(self):
        from run import sha
        if not (HERE/'frozen_config.json').exists():self.skipTest('freeze not yet run')
        cfg=json.loads((HERE/'frozen_config.json').read_text())
        self.assertEqual(sha(HERE/'frozen_config.json'),(HERE/'frozen_config.sha256').read_text().strip())
        self.assertEqual(cfg['beta'],.30)
        for name,digest in cfg['provenance']['source_hashes'].items():self.assertEqual(sha(HERE/name),digest)
        self.assertEqual(cfg['split_sha256'],sha(HERE/'split.json'))
        if not (HERE/'input_validation.json').exists():self.skipTest('independent input validation not yet run')
        val=json.loads((HERE/'input_validation.json').read_text())
        self.assertEqual(len(val['patches']),20)
        self.assertTrue(all(x['gt_crop_exact'] and x['rgb_lossless'] for x in val['patches'].values()))
    def test_probability_unary(self):
        p=np.array([[[.5,.25,.125,.0625,.0625,0.]]])
        u=unary(p)
        self.assertEqual(u[0,0,0],0)
        self.assertAlmostEqual(u[0,0,1],np.log(2))
        self.assertAlmostEqual(u[0,0,5],-np.log(1e-12)+np.log(.5))
    def test_real_outputs(self):
        p=HERE/'results.json'
        if not p.exists():self.skipTest('evaluate not yet run')
        result=json.loads(p.read_text())
        self.assertEqual(result['changed_pixels'],result['wrong_to_correct']+result['correct_to_wrong']+result['wrong_to_different_wrong'])
        for row in result['patches']:
            self.assertTrue(row['beta_zero_exact']);self.assertTrue(row['solver']['nonincreasing_energy'])
            i=row['image_index'];z=np.load(HERE/f'cache/tp_{i:04d}.npz');validate_probabilities(z['probabilities'])
            with np.load(HERE/f'predictions/patch_{i:04d}.npz') as a:np.testing.assert_array_equal(a['baseline'],z['probabilities'].argmax(-1))

if __name__=='__main__':unittest.main()
