"""Regression checks for reconstructed room surfaces and coordinate preservation."""
import unittest
import numpy as np
from room_views import create_tensor_volume, integrate_tensor_frame, aligned_coordinates, o3d

class RoomRegressionTests(unittest.TestCase):
    def test_metric_plane_produces_colored_mesh(self):
        # This known plane failed with the legacy TSDF backend in Open3D 0.20.
        volume=create_tensor_volume(.025)
        depth=np.full((120,160),2000,np.uint16)
        rgb=np.full((120,160,3),[210,90,40],np.uint8)
        intrinsic=o3d.core.Tensor([[120.,0,79.5],[0,120.,59.5],[0,0,1]],o3d.core.float64)
        for offset in [0.,.03,-.03]:
            extrinsic=np.eye(4);extrinsic[0,3]=offset
            self.assertTrue(integrate_tensor_frame(volume,depth,rgb,intrinsic,extrinsic))
        mesh=volume.extract_triangle_mesh(weight_threshold=1.).to_legacy()
        vertices=np.asarray(mesh.vertices);colors=np.asarray(mesh.vertex_colors)
        self.assertGreater(len(vertices),100)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertLess(abs(np.median(vertices[:,2])-2.),.04)
        self.assertEqual(colors.shape,vertices.shape)
        self.assertGreater(colors[:,0].mean(),colors[:,2].mean())

    def test_empty_supported_frame_is_skipped(self):
        volume=create_tensor_volume()
        k=o3d.core.Tensor(np.eye(3),o3d.core.float64)
        self.assertFalse(integrate_tensor_frame(volume,np.zeros((32,32),np.uint16),np.zeros((32,32,3),np.uint8),k,np.eye(4)))

    def test_display_alignment_preserves_distances_and_floor(self):
        angle=.43;A=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        original=np.array([[1.,-.8,2.],[2.,.2,-1.],[-1.,1.,0.]])
        aligned=aligned_coordinates(original,A,-.8)
        recovered=np.column_stack([(aligned[:,:2]@A.T)[:,0],aligned[:,2]-.8,(aligned[:,:2]@A.T)[:,1]])
        np.testing.assert_allclose(recovered,original,atol=1e-12)
        self.assertEqual(aligned[0,2],0.)
        np.testing.assert_allclose(np.linalg.norm(np.diff(original,axis=0),axis=1),np.linalg.norm(np.diff(aligned,axis=0),axis=1))

if __name__=='__main__':unittest.main()
