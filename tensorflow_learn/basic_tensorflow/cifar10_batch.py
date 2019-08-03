import tensorflow as tf
import os
data_dir='C:/Users/25781/Desktop/cifar10_2'

def _read_input(filename_queue):
  label_bytes = 1
  height = 32
  depth = 3
  image_bytes = height * height * depth
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, byte_data = reader.read(filename_queue)
  uint_data = tf.decode_raw(byte_data, tf.uint8)
  label = tf.cast(tf.strided_slice(uint_data, [0], [label_bytes]), tf.int32)
  label.set_shape([1])
  depth_major = tf.reshape(
      tf.strided_slice(uint_data, [label_bytes], [record_bytes]),
      [depth, height, height])
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  return image, label
def _distort_resize(image, image_size):
  distorted_image = tf.random_crop(image, [image_size, image_size, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
  distorted_image.set_shape([image_size, image_size, 3])
  return distorted_image
def _batch_features(image, label, batch_size, split, image_size):
  image = tf.transpose(image, [2, 0, 1])
  if split == 'train':
    batched_features = tf.train.shuffle_batch(
        [image,label],
        batch_size=batch_size,
        num_threads=16,
        capacity=10000 + 3 * batch_size,
        min_after_dequeue=10000)
  else:
    batched_features = tf.train.batch(
        [image,label],
        batch_size=batch_size,
        num_threads=1,
        capacity=10000 + 3 * batch_size)
  return batched_features
def inputs(split, batch_size):
  if split == 'train':
    filenames = ['C:/Users/25781/Desktop/cifar10_2/data_batch_%d.bin' % i for i in range(1, 6)]
  else:
    filenames = ['C:/Users/25781/Desktop/cifar10_2/test_batch.bin']
  filename_queue = tf.train.string_input_producer(filenames)
  float_image, label = _read_input(filename_queue)
  image_size = 24
  if split == 'train':
    resized_image = _distort_resize(float_image, image_size)
  else:
    resized_image = tf.image.resize_image_with_crop_or_pad(
        float_image, image_size, image_size)
  image = tf.image.per_image_standardization(resized_image)
  return _batch_features(image, label, batch_size, split, image_size)
test=inputs('train',1000)
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  print(sess.run(test))
  #coord.request_stop()
  #for thread in threads:
     #thread.join()
  
