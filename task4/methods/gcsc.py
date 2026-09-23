"""GCSC (manual §4 step 3): identical objective to Vanilla. The single change, RandAugment(2, 9) after
crop/flip, lives in the data pipeline (configs/gcsc.yaml -> data.randaugment -> data/cifar10.train_transform)."""
from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    name = "gcsc"
