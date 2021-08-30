"""
convert midi files to Note_Seq proto files and write TFRecord file
"""
from absl import flags
from absl import logging
from absl import app
import hashlib
import os
import tensorflow as tf
from note_seq import midi_io

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'directory of input MIDI file')
flags.DEFINE_string('output_file', None, 'path of output TFRecord file')
flags.DEFINE_string('log', 'INFO', 'what message will be logged'
                                   'DEBUG, INFO, WARN, ERROR, or FATAL.')


def generate_note_sequence_id(filename, collection_name):
    """Generates a unique ID for a sequence.

      The format is:'/id/<collection name>/<hash>'.

      Args:
        filename: The string path to the source file relative to the root of the
            collection.
        collection_name: The collection from which the file comes.

      Returns:
        The generated sequence ID as a string.
      """
    filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
    return '/id/%s/%s' % (collection_name, filename_fingerprint.hexdigest())


def convert_files(convert_dir, writer):
    """Converts files.

      Args:
        root_dir: A string specifying a root directory.
        writer: A TFRecord writer

      Returns:
        A map from the resulting Futures to the file paths being converted.
      """
    files_to_convert = tf.io.gfile.listdir(convert_dir)
    logging.info("Converting files in %s" % convert_dir)
    for file_to_convert in files_to_convert:
        full_file_path = os.path.join(convert_dir, file_to_convert)
        if full_file_path.lower().endswith('.mid') or full_file_path.lower().endswith('.midi'):
            try:
                sequence = convert_midi(convert_dir, full_file_path)
            except Exception as e:
                logging.fatal("%s generated an exception %s" % (full_file_path, e))
                continue
            if sequence:
                writer.write(sequence.SerializeToString())
        elif full_file_path.lower().endswith('//'):
            print('folder')
        else:
            logging.warning('Unable to find a converter for file %s', full_file_path)


def convert_midi(root_dir, full_file_path):
    """Converts a midi file to a sequence proto.

      Args:
        root_dir: A string specifying the root directory for the files being
            converted.
        full_file_path: the full path to the file to convert.

      Returns:
        Either a NoteSequence proto or None if the file could not be converted.
      """
    try:
        sequence = midi_io.midi_to_note_sequence(
            tf.io.gfile.GFile(full_file_path, 'rb').read())
    except midi_io.MIDIConversionError as e:
        logging.warning('MIDI file %s could not be converted' % full_file_path)
        return None
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.basename(full_file_path)
    sequence.id = generate_note_sequence_id(sequence.filename, sequence.collection_name)
    logging.info('Converted MIDI file %s' %full_file_path)
    return sequence


def convert_directory(root_dir, output_file):
    """Converts files to NoteSequences and writes to `output_file`.

  Input files found in `root_dir` are converted to NoteSequence protos with the
  basename of `root_dir` as the collection_name, and the relative path to the
  file from `root_dir` as the filename.

  Args:
    root_dir: A string specifying a root directory.
    output_file: Path to TFRecord file to write results to.
  """
    with tf.io.TFRecordWriter(output_file) as writer:
        convert_files(root_dir, writer)


def main(unused_argv):
    logging.set_verbosity(FLAGS.log)
    if not FLAGS.input_dir:
        logging.fatal('--input_dir needed')
    if not FLAGS.output_file:
        logging.fatal('--output_file needed')

    input_dir = FLAGS.input_dir
    output_file = FLAGS.output_file
    output_dir = os.path.dirname(output_file)

    if output_dir:
        tf.io.gfile.makedirs(output_dir)

    convert_directory(input_dir, output_file)



# def console_entry_point():



if __name__ == '__main__':
    app.run(main)
    # console_entry_point()
