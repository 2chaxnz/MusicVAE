import contrib.rnn as contrib_rnn
import tensorflow.compat.v1 as tf

rnn = tf.nn.rnn_cell


def rnn_cell(rnn_cell_size, dropout_keep_prob, is_training=True):
    cells = []
    dropout_keep_prob = dropout_keep_prob if is_training else 1.0  # 1.0 for eval
    for i in range(len(rnn_cell_size)):
        cell = contrib_rnn.LSTMBlockCell(rnn_cell_size[i])
        cell = rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob)
        cells.append(cell)
    return rnn.MultiRNNCell(cells)


def build_bidirectional_lstm(layer_sizes, dropout_keep_prob, is_training=True):
    cells_fw = []
    cells_bw = []
    for layer_size in layer_sizes:
        cells_fw.append(rnn_cell([layer_size], dropout_keep_prob, is_training))
        cells_bw.append(rnn_cell([layer_size], dropout_keep_prob, is_training))
    return cells_fw, cells_bw










