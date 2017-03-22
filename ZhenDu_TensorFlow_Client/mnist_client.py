# encoding: utf-8
#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf
import gzip

import struct
import ctypes

from PIL import Image

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.example import mnist_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      print("预测的数字",str(prediction))
      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  test_data_set = mnist_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)
  for _ in range(num_tests):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    image, label = test_data_set.next_batch(1)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(label[0], result_counter))
  return result_counter.get_error_rate()

def ImageToMatrix(filename):
  # 读取图片
  im = Image.open(filename)
  # 显示图片
#im.show()
  width,height = im.size
  im = im.convert("L")
  data = im.getdata()
  data = numpy.matrix(data,dtype='float')
  #new_data = np.reshape(data,(width,height))
  new_data = numpy.reshape(data,(height,width))
  return new_data

def main(_):
  filename = raw_input("请输入要判别的文件:")
  data = ImageToMatrix(filename)
  #创建IDX文件
  offset = 0
  fmt_header = '>iiii'
  img_size = 28 * 28
  fmt_img = '>' + str(img_size) + 'B'
  # print struct.calcsize(fmt_header)
  buf = ctypes.create_string_buffer(struct.calcsize(fmt_header)+struct.calcsize(fmt_img))

  #写入描述训练集的第一行
  struct.pack_into(fmt_header,buf , offset ,2051 , 1 , 28 , 28)
  #偏移量按字节算
  offset = offset+struct.calcsize(fmt_header)
  
  #然后写入一张图片
  data = data.reshape(1,img_size)
  
  #将线性矩阵转化为数组
  arr = data.getA1()

  struct.pack_into(fmt_img, buf, offset , *arr)
  # print binascii.hexlify(buf)
  with open("/tmp/t10k-images-idx3-ubyte","wb") as f:
    f.write(buf)
  with open('/tmp/t10k-images-idx3-ubyte', 'rb') as fRead:
    with gzip.open('/tmp/t10k-images-idx3-ubyte.gz', 'wb') as f:
      f.writelines(fRead)
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,
                            FLAGS.concurrency, FLAGS.num_tests)
  #print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()
