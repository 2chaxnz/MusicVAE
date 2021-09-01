
from absl import logging
import tensorflow as tf
import contrib.rnn as contrib_rnn
import contrib.training as contrib_training
import lstm_utils


class BidirectionalLstmEncoder():
    """Bidirectional LSTM Encoder.
    abstract class and decorator used in original code but not use in the case
    """

    def output_depth(self):

        """Returns the size of the output final dimension."""
        return self._cells[0][-1].output_size + self._cells[1][-1].output_size

    def build(self, hparams, is_training=True):
        """Builder method for BaseEncoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      is_training: Whether or not the model is being used for training.
    """
        self._is_training = is_training
        logging.info("\nEncoder Cells (bidirectional):\n"
                     "  units: [%d]" %hparams.enc_rnn_size[0])
        self._cells = lstm_utils.build_bidirectional_lstm(
            hparams.enc_rnn_size,
            hparams.dropout_keep_prob,
            is_training
        )


    def encode(self, sequence, sequence_length):
        """Encodes input sequences into a precursors for latent code `z`.

    Args:
       sequence: Batch of sequences to encode.
       sequence_length: Length of sequences in input batch.

    Returns:
       outputs: Raw outputs to parameterize the prior distribution in
          MusicVae.encode, sized `[batch_size, N]`.
    """
        cells_fw, cells_bw = self._cells
        _, output_state_fw, output_state_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, sequence, dtype=tf.float32, sequence_length=sequence_length)

        last_state_fw = output_state_fw[-1][-1].h
        last_state_bw = output_state_bw[-1][-1].h
        return tf.concat([last_state_fw,last_state_bw],1)


def get_default_hparams():
    """Returns copy of default HParams for LSTM models."""
    hparams_map = {
        'conditional': True,
        'dec_rnn_size': [512],  # Decoder RNN: number of units per layer.
        'enc_rnn_size': [256],  # Encoder RNN: number of units per layer per dir.
        'dropout_keep_prob': 1.0,  # Probability all dropout keep.
        'sampling_schedule': 'constant',  # constant, exponential, inverse_sigmoid
        'sampling_rate': 0.0,  # Interpretation is based on `sampling_schedule`.
        'use_cudnn': False,  # DEPRECATED
        'residual_encoder': False,  # Use residual connections in encoder.
        'residual_decoder': False,  # Use residual connections in decoder.
        'control_preprocessing_rnn_size': [256],  # Decoder control preprocessing.
        'max_seq_len': 64,
        'z_size': 256,
        'free_bits': 48,
        'max_beta': 0.2,
        'clip_mode': 'global_norm',
        'grad_norm_clip_to_zero': 10000,
        'learning_rate': 0.001,
        'decay_rate': 0.9999,
        'min_learning_rate': 0.00001,
    }
    return contrib_training.HParams(**hparams_map)



