import json
from pathlib import Path
import unittest
import numpy as np
from ester_oil.__main__ import ROOT
from ester_oil.selection import select

class Contracts(unittest.TestCase):
    def test_configs_are_portable(self):
        registry=json.loads((ROOT/'configs/experiments.json').read_text())
        self.assertEqual(len(registry),13)
        for name,spec in registry.items():
            conf=json.loads((ROOT/'configs/qsar'/f'{name}.json').read_text())
            self.assertFalse(Path(conf['data_path']).is_absolute())
            self.assertEqual(conf['seed'],2026)
            self.assertEqual(conf['data_path'],spec['data'])

    def test_feasibility_precedes_objectives(self):
        indices=select(np.array([[0.,0.],[10.,10.],[20.,20.]]),1,np.array([[1.],[-1.],[-1.]]))
        self.assertEqual(indices.tolist(),[1])

    def test_exact_front_boundary_and_ties(self):
        f=np.array([[0.,1.],[1.,0.],[2.,2.]])
        self.assertEqual(select(f,2).tolist(),[0,1])
        self.assertEqual(select(np.ones((4,3)),3).tolist(),[0,1,2])

    def test_objective_direction(self):
        self.assertEqual(select(np.array([[3.,4.],[1.,2.],[5.,6.]]),1).tolist(),[1])

if __name__=='__main__':unittest.main()
