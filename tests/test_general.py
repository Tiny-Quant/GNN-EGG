import unittest

import sys
from os.path import dirname, abspath
repo_dir = dirname(dirname(abspath(__file__)))
sys.path.append(repo_dir)

from egg_models import egg_generic

class TestEggGeneric(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_generator = egg_generic.EggGeneric(
            max_node_size=10, 
            cont_node_feats=10, 
            cont_edge_feats=10, 
            dis_node_feats=(1, 2, 3), 
            dis_edge_feats=(1, 2), 
            batch_size=2
        )

        self.gen = self.mock_generator()
    
    def test_shapes(self):
        self.assertEqual(
            self.gen['cont_node_feats'].shape, 
            (2, 10, 10)
        )
        self.assertEqual(
            self.gen['cont_edge_feats'].shape, 
            (2, 10**2, 10)
        )
        self.assertEqual(
            self.gen['dis_node_feats'].shape, 
            (2, 10, 1 + 2 + 3)
        )
        self.assertEqual(
            self.gen['dis_edge_feats'].shape, 
            (2, 100, 1 + 2)
        )
        self.assertEqual(
            self.gen['full_edge_indices'].shape, 
            (2, 2, 10**2)
        )
        self.assertEqual(
            self.gen['adjacency_matrix'].shape, 
            (2, 10, 10)
        )
        self.assertEqual(
            self.gen['C_x_logLik'].shape, 
            (2, 3)
        )
        self.assertEqual(
            self.gen['C_e_logLik'].shape, 
            (2, 2)
        )
        self.assertEqual(
            self.gen['A_logLik'].shape, 
            (2,)
        )

if __name__ == '__main__':
    unittest.main()