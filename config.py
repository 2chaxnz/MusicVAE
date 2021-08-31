''' Configuration for the model'''

import collections
from common import merge_hparams
from contrib import training as contrib_training

import data
import lstm_models
from base_model import MusicVAE

HParams = contrib_training.HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path', 'tfds_name'])):

    def values(self):
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)

CONFIG_MAP = {}

CONFIG_MAP['drum_4bar'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()), #4-bar: short sequence/ flat decoder
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 4,  # 4 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=4,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)
