import pytest
import importlib
import inspect
import config


def test_output_dir():
    """check if the expected methods of predictors exist in configured predictors"""
    for cam_name, cam_dict in config.cameras.items():
        assert 'output_dir' in cam_dict
        assert cam_dict['output_dir'] is None


def test_cam_packages_exist():
    cam_modules = set([cam_dict['module'] for cam_dict in config.cameras.values()])
    for cm in cam_modules:
        if cm == 'flir':
            import PySpin
        elif cm == 'allied_vision':
            import vimba