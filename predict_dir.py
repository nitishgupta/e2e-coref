from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  # Input directory containing files in .jsonlines format.
  input_dirname = sys.argv[2]

  # Predictions will be written to this directory in .jsonlines format.
  output_dirname = sys.argv[3]

  model = cm.CorefModel(config)

  with tf.Session() as session:
    model.restore(session)

    filenames = [f for f in listdir(input_dirname) if isfile(join(input_dirname, f))]

    for filename in filenames:
      input_filename = os.path.join(input_dirname, filename)
      output_filename = os.path.join(output_dirname, filename)

      with open(output_filename, "w") as output_file:
        with open(input_filename) as input_file:
          for example_num, line in enumerate(input_file.readlines()):
            example = json.loads(line)
            tensorized_example = model.tensorize_example(example, is_training=False)
            feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
            _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
            predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

            output_file.write(json.dumps(example))
            output_file.write("\n")
            if example_num % 100 == 0:
              print("Decoded {} examples.".format(example_num + 1))
