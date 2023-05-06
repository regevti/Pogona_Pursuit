import pytest
import json
from pathlib import Path


periphery_config_path = Path('configurations/periphery_config.json')


def test_periphery_config_exists():
    assert periphery_config_path.exists()


def test_structure():
    with periphery_config_path.open('r') as f:
        cfg = json.load(f)

    assert 'arena' in cfg
    # assert 'camera_trigger' in cfg
    for i, device in enumerate(cfg['arena']['interfaces']):
        assert 'name' in device, f'Device #{i + 1} has no name'
        device_name = device['name']
        assert 'type' in device, f'Device {device_name} has no "type"'
        assert 'pins' in device or 'pin' in device, f'Device {device_name} has no "pin" or "pins"'
        if device['type'] == 'feeder':
            assert 'order' in device, f'Feeder {device["name"]} must have "order" field'
