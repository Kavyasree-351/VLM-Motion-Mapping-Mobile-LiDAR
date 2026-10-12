"""Regression tests for the causes that invalidated the original maps."""
import unittest
import numpy as np
from geometry_probe import cameras, intrinsic, triangulate, epipolar, projection
from improve_mapping import fit_depth, predict_depth, voxel_fuse
from motion_map import residual_covariance,motion_statistics,parse_motion_query,execute_attribute_query

class MappingRegressionTests(unittest.TestCase):
    def test_arkit_axes_and_triangulation(self):
        poses=np.array([[0,0,0,0,0,0,1],[.2,.03,0,0,np.sin(.08),0,np.cos(.08)]],float)
        R,t=cameras(poses,'c2w_opengl');K=intrinsic()
        points=np.array([[.1,.2,-2],[.4,-.1,-3],[-.2,.1,-1.5]])
        def image(i):
            p=(points-t[i])@R[i]@K.T
            return p[:,:2]/p[:,2:]
        a,b=image(0),image(1)
        xyz,z1,z2,e,angle=triangulate(a,b,R[0],t[0],R[1],t[1],K)
        np.testing.assert_allclose(xyz,points,atol=1e-8)
        self.assertTrue(np.all(z1>0));self.assertTrue(np.all(z2>0))
        self.assertLess(epipolar(a,b,R[0],t[0],R[1],t[1],K).max(),1e-8)
        wrongR,wrongt=cameras(poses,'c2w_opencv')
        wrongxyz,_,_,_,_=triangulate(a,b,wrongR[0],wrongt[0],wrongR[1],wrongt[1],K)
        self.assertFalse(np.allclose(wrongxyz,points,atol=.1),'Original convention must fail world geometry.')
        self.assertGreater(epipolar(a,b,wrongR[0],wrongt[0],wrongR[1],wrongt[1],K).mean(),1.)

    def test_temporal_calibration_recovers_known_scale(self):
        rng=np.random.default_rng(42);samples=[]
        for frame in range(0,2091,20):
            a=1.3+.2*np.sin(frame/400);b=.9+.1*np.cos(frame/600)
            for raw in np.linspace(5,250,30):
                z=a*raw/255+b+rng.normal(0,.003)
                samples.append([frame,frame+20,0,0,raw,z,0,5,0,0,0])
        s=np.array(samples);train=np.arange(len(s))%3!=0
        model=fit_depth(s,selected=train)
        pred=predict_depth(model,s[~train,0],s[~train,4])
        self.assertLess(np.median(abs(pred-s[~train,5])/s[~train,5]),.01)

    def test_one_view_cannot_create_multi_view_support(self):
        points=np.tile([[.001,.002,.003]],(100,1));colors=np.ones_like(points)*.5
        _,_,views,_=voxel_fuse(points,colors,np.zeros(len(points),int),np.ones(len(points)))
        np.testing.assert_array_equal(views,[1])
        _,_,views,_=voxel_fuse(points,colors,np.arange(len(points))%2,np.ones(len(points)))
        np.testing.assert_array_equal(views,[2])

    def test_weak_frames_cannot_reverse_depth_order(self):
        samples=[]
        # Deliberately contradictory anchors would produce a negative slope
        # with the original unconstrained least-squares decoder.
        for f in [0,20,40]:
            for raw in np.linspace(5,250,12):
                samples.append([f,f+20,0,0,raw,3-raw/255,0,5,0,0,0])
        s=np.asarray(samples)
        m=fit_depth(s)
        self.assertTrue(np.all(m['coeff'][:,0]>0))
        self.assertTrue(np.all(predict_depth(m,0,np.array([0,255]))>0))
        self.assertGreater(predict_depth(m,0,255),predict_depth(m,0,0))

    def test_uniform_duplicate_pixels_do_not_change_fusion(self):
        points=np.array([[.005,.01,.015],[.01,.01,.015],[.015,.01,.015]])
        colors=np.array([[1,0,0],[0,1,0],[0,0,1]],float);ids=np.array([0,0,20]);weights=np.array([.8,.8,.7])
        a=voxel_fuse(points,colors,ids,weights)
        b=voxel_fuse(np.repeat(points,10,axis=0),np.repeat(colors,10,axis=0),np.repeat(ids,10),np.repeat(weights,10))
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-8)

    def test_vlmm_v2_covariance_rotates_source_endpoint(self):
        a=np.diag([.01,.04,.09]);b=np.diag([.02,.03,.07]);R=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        expected=b+R@a@R.T
        np.testing.assert_allclose(residual_covariance(a,b,R,pose_floor=0),expected)
        self.assertFalse(np.allclose(expected,a+R@b@R.T),'v1 endpoint swap must not pass.')

    def test_motion_noise_and_true_displacement_are_distinct(self):
        rng=np.random.default_rng(77);cov=np.tile(np.eye(3)*.02**2,(10000,1,1))
        static=rng.normal(0,.02,(10000,3));moving=static+[.30,0,0]
        quiet=motion_statistics(static,cov);active=motion_statistics(moving,cov)
        self.assertGreater(quiet['fraction_flagged'],.03);self.assertLess(quiet['fraction_flagged'],.07)
        self.assertGreater(active['score'],2.8);self.assertGreater(active['reliability'],.9)

    def test_motion_prior_does_not_answer_observed_motion(self):
        e={'id':'chair','class_name':'chair','world_x':0.,'world_z':0.,'movability_prior_score':.95,'motion_class':'movable_not_observed_moving','motion_reliability_score':0.,'reobservation_priority':1.}
        self.assertEqual(len(execute_attribute_query('what could be moved?',[e])['matches']),1)
        self.assertEqual(len(execute_attribute_query('what was observed moving?',[e])['matches']),0)
        self.assertEqual(parse_motion_query('what is not moving?')['intent'],'observed_static')
        self.assertEqual(execute_attribute_query('what is moving now?',[e])['status'],'needs_fresh_observation')
        self.assertEqual(parse_motion_query('moving and portable chairs')['intent'],'unsupported_composition')

    def test_decoder_uncertainty_suppresses_axial_noise_not_lateral_motion(self):
        n=1000;noise=np.tile([0.,0.,.08],(n,1))
        sensor=np.tile(np.diag([.01**2,.01**2,.015**2]),(n,1,1))
        augmented=sensor+np.diag([0.,0.,.12**2])
        raw=motion_statistics(noise,sensor);repaired=motion_statistics(noise,augmented)
        true_move=motion_statistics(noise+[.25,0,0],augmented)
        self.assertGreater(raw['score'],2.8);self.assertLess(repaired['score'],2.8)
        self.assertGreater(true_move['score'],2.8);self.assertGreater(true_move['reliability'],.9)

if __name__=='__main__':unittest.main(verbosity=2)
