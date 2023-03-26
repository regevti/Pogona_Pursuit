import yaml
from pathlib import Path


class Predictor:
    def __init__(self):
        self.pred_config = dict()
        self.load_pred_config()
        self.threshold = self.pred_config['threshold']
        self.model_path = self.pred_config['model_path']

    def predict(self, frame, timestamp):
        raise NotImplemented('No predict method')

    def load_pred_config(self):
        pconfig = yaml.load(Path('configurations/predict_config.yaml').open(), Loader=yaml.FullLoader)
        predictor_name = type(self).__name__
        for k, d in pconfig.items():
            if d.get('predictor_name') == predictor_name:
                self.pred_config = d
                break
        assert self.pred_config, f'Could not find config for {predictor_name}'
