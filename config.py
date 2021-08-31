''' Configuration for the model'''

import collections
from common import merge_hparams
from contrib import training as contrib_training

#import 구현후 수정
# from magenta.models.music_vae import data
# from magenta.models.music_vae import lstm_models
# from magenta.models.music_vae.base_model import MusicVAE

HParams = contrib_training.HParams



class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

  def values(self):
    return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)

CONFIG_MAP = {}

