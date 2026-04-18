from src.preprocess import run_preprocess
from src.train import run_training
from src.register import run_registry
from src.export_model import export_model


def test_preprocess():
    assert run_preprocess() == "SUCCESS"


def test_train():
    assert run_training() == "TRAINING_DONE"


def test_registry():
    assert run_registry() == "REGISTERED"


def test_export():
    assert export_model() == "EXPORTED"