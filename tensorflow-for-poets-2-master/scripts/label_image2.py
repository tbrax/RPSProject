# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import datetime
import cv2

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
	file_name = "tf_files/test_photos/test.jpg"
	model_file = "tf_files/retrained_graph.pb"
	label_file = "tf_files/retrained_labels.txt"
	input_height = 224
	input_width = 224
	input_mean = 128
	input_std = 128
	input_layer = "input"
	output_layer = "final_result"
	device = 0
	cap = cv2.VideoCapture(device)

	#MMMMMMMMMMMMMMMMMMMMMM
	if not cap.isOpened():
		cap.open(device)

	# only attempt to read if it is opened
	if cap.isOpened:
		while True:
			re, img = cap.read()
			# Only display the image if it is not empty
			if re:
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

				# define range of blue color in HSV
				lower_blue = np.array([110,50,50])
				upper_blue = np.array([130,255,255])

				# Threshold the HSV image to get only blue colors
				mask = cv2.inRange(hsv, lower_blue, upper_blue)
				res = cv2.bitwise_and(img,img, mask= mask)

				cv2.imshow("video output", img)
				k2 = cv2.waitKey(10) & 0xFF
				if k2 == 32:
					
					timetup = time.gmtime()
					now = time.strftime('%Y %m %d %H %M %S', timetup)
					cv2.imwrite( "tf_files\\test_photos\\test.jpg", img)
					graph = load_graph(model_file)
					t = read_tensor_from_image_file(file_name,
							  input_height=input_height,
							  input_width=input_width,
							  input_mean=input_mean,
							  input_std=input_std)
					time.sleep(0.7)

					input_name = "import/" + input_layer
					output_name = "import/" + output_layer
					input_operation = graph.get_operation_by_name(input_name);
					output_operation = graph.get_operation_by_name(output_name);

					with tf.Session(graph=graph) as sess:
						start = time.time()
						results = sess.run(output_operation.outputs[0],
										  {input_operation.outputs[0]: t})
						end=time.time()
					results = np.squeeze(results)

					top_k = results.argsort()[-5:][::-1]
					labels = load_labels(label_file)

					print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
					template = "{} (score={:0.5f})"
					for i in top_k:
						print(template.format(labels[i], results[i]))

			# if it is empty abort
			else:
				print("Error reading capture device")
				break
			k = cv2.waitKey(10) & 0xFF
			if k == 27:
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
		print("Failed to open capture device")
	
	#MMMMMMMMMMMMMMMMMMMMMMMM
	


