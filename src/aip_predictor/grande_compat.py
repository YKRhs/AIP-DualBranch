from __future__ import annotations

import sys
import types


class DummyLabelEncoderFeatureGenerator:
    def __init__(self, *args, **kwargs):
        self.features_in = []

    def fit(self, X=None, **kwargs):
        self.features_in = []
        return self

    def transform(self, X=None, **kwargs):
        return X


def install_grande_compatibility() -> None:
    if "autogluon.features.generators" not in sys.modules:
        autogluon = types.ModuleType("autogluon")
        features = types.ModuleType("autogluon.features")
        generators = types.ModuleType("autogluon.features.generators")
        generators.LabelEncoderFeatureGenerator = DummyLabelEncoderFeatureGenerator
        sys.modules.setdefault("autogluon", autogluon)
        sys.modules.setdefault("autogluon.features", features)
        sys.modules.setdefault("autogluon.features.generators", generators)

    import torch

    if not getattr(torch, "_aip_grande_compile_disabled", False):
        torch.compile = lambda model, *args, **kwargs: model
        torch._aip_grande_compile_disabled = True


def get_grande_class():
    install_grande_compatibility()
    from GRANDE import GRANDE

    return GRANDE
