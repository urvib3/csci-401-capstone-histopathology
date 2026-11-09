import unittest
import numpy as np
from core import costs,initial_unary,SIZES,graph,energy

class Checks(unittest.TestCase):
    def test_scale_factors(self):
        tabs=np.stack([1-np.eye(6)]*6);a=costs(tabs,.3,.1,False);b=costs(tabs,.3,.1,True)
        for l,s in enumerate(SIZES):np.testing.assert_allclose(b[l],a[l]*s)
        for l,s in enumerate(SIZES[:-1]):np.testing.assert_allclose(b[6+l],a[6+l]*s*s)
        p=np.random.default_rng(2).dirichlet(np.ones(6),size=(32,32));y,u=initial_unary(p,.3,False);z,v=initial_unary(p,.3,True);np.testing.assert_array_equal(y,z);g=graph(32)
        for l,s in enumerate(SIZES):
            sl=slice(g['offsets'][l],g['offsets'][l+1]);np.testing.assert_allclose(v[sl],u[sl]*(s*s),atol=1e-12)

    def test_zero_siblings(self):
        c=costs(np.zeros((6,6,6)),.3,.1);self.assertTrue((c[:6]==0).all());self.assertTrue((c[6:,0,1]==.1).all())

if __name__=='__main__':unittest.main()
