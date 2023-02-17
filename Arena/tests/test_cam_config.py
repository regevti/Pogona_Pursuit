import pytest
import importlib
import inspect
import config


def test_output_dir():
    """check if the expected methods of predictors exist in configured predictors"""
    for cam_name, cam_dict in config.cameras.items():
        assert 'output_dir' in cam_dict
        assert cam_dict['output_dir'] is None


