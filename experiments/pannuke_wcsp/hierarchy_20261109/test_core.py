"""Independent checks for graph structure and energy accounting."""
import unittest
import numpy as np
from core import *

class Checks(unittest.TestCase):
    def test_graph(self):
        g=graph()
        self.assertEqual(g['offsets'][-1],87360)
        self.assertEqual(sum(x['parent_child_edges'] for x in g['counts']),87296)
        self.assertTrue(np.all(g['a']!=g['b']))
        pairs=np.sort(np.stack([g['a'],g['b']],1),1)
        self.assertEqual(len(np.unique(pairs,axis=0)),len(pairs))
        self.assertEqual(g['ptr'][-1],2*len(g['a']))

    def test_independent_energy(self):
        rng=np.random.default_rng(2);g=graph(32);p=rng.dirichlet(np.ones(6),size=(32,32))
        y,u=initial_unary(p,.3);c=costs(np.stack([1-np.eye(6)]*6),.3,.1)
        slow=sum(u[i,int(k)] for i,k in enumerate(y))
        slow+=sum(c[int(t),int(y[a]),int(y[b])] for a,b,t in zip(g['a'],g['b'],g['t']))
        self.assertAlmostEqual(energy(y,u,c,g),slow,places=8)
        # Check exact energy change against a manually computed local change.
        for i in [0,32,1023,1024,len(y)-1]:
            z=y.copy();z[i]=(y[i]+1)%6
            delta=u[i,z[i]]-u[i,y[i]]
            for a,b,t in zip(g['a'],g['b'],g['t']):
                if a==i or b==i:delta+=c[t,z[a],z[b]]-c[t,y[a],y[b]]
            self.assertAlmostEqual(energy(z,u,c,g)-energy(y,u,c,g),delta,places=8)

    def test_unary_toggle_and_zero_equivalence(self):
        rng=np.random.default_rng(3);p=rng.dirichlet(np.ones(6),size=(32,32));g=graph(32)
        y,a=initial_unary(p,0);z,b=initial_unary(p,.3)
        np.testing.assert_array_equal(y,z);np.testing.assert_array_equal(a[:1024],b[:1024]);self.assertTrue((a[1024:]==0).all())
        pred,info=solve(p,np.stack([1-np.eye(6)]*6),g,0,0)
        np.testing.assert_array_equal(pred[:1024],p.argmax(-1).ravel());self.assertTrue(info['verified_local_minimum'])

    def test_table_and_majority(self):
        gt=np.array([[1,1],[2,0]],np.uint8);self.assertEqual(majority(gt,2)[0,0],1)
        n=counts(np.array([[0,1],[2,3]],np.uint8));self.assertEqual(n.sum(),8)
        tab,ev=fit_table(n);np.testing.assert_allclose(tab,tab.T);self.assertTrue((tab>=0).all());self.assertTrue((np.diag(tab)==0).all())
        self.assertAlmostEqual(tab[~np.eye(6,dtype=bool)].mean(),1)

    def test_solver_audit(self):
        p=np.random.default_rng(7).dirichlet(np.ones(6),size=(32,32))
        y,info=solve(p,np.stack([1-np.eye(6)]*6),graph(32),.3,.1,.3)
        self.assertTrue(np.all(np.diff(info['energy'])<=1e-7));self.assertTrue(info['verified_local_minimum'])

if __name__=='__main__':unittest.main()
