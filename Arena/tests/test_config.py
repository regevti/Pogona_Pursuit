import pytest
import importlib
import inspect
import config


def test_predictors():
    """check if the expected methods of predictors exist in configured predictors"""
    methods2check = ['predict']
    for prd_name, (prd_module, prd_class) in config.arena_modules['predictors'].items():
        prd_module = importlib.import_module(prd_module)
        predictor = getattr(prd_module, prd_class)
        for m in methods2check:
            assert getattr(predictor, m)

