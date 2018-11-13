from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import tempfile

import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions, SetupOptions
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as tft_beam
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from datetime import datetime


def main():
  def preprocessing_fn(inputs):
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']
    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    s_integerized = tft.compute_and_apply_vocabulary(s)
    x_centered_times_y_normalized = (x_centered * y_normalized)
    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        'x_centered_times_y_normalized': x_centered_times_y_normalized,
        's_integerized': s_integerized
    }

  raw_data = [
      {'x': 1, 'y': 1, 's': 'hello'},
      {'x': 2, 'y': 2, 's': 'world'},
      {'x': 3, 'y': 3, 's': 'hello'}
  ]

  raw_data_metadata = dataset_metadata.DatasetMetadata(
      dataset_schema.from_feature_spec({
          's': tf.FixedLenFeature([], tf.string),
          'y': tf.FixedLenFeature([], tf.float32),
          'x': tf.FixedLenFeature([], tf.float32),
      }))

  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DataflowRunner'
  google_cloud_options = options.view_as(GoogleCloudOptions)
  google_cloud_options.project = 'PROJECT-NAME'
  google_cloud_options.staging_location = 'gs://BUCKET-NAME/staging'
  google_cloud_options.temp_location = 'gs://BUCKET-NAME/temp'
  google_cloud_options.job_name = 'JOBNAME-USERNAME-' + datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S%z')
  options.view_as(SetupOptions).requirements_file = 'requirements.txt'
  with beam.Pipeline(options=options) as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      transform_fn = pipeline | transform_fn_io.ReadTransformFn('gs://BUCKET-NAME/transform_output')
      transformed_dataset = (
          ((raw_data, raw_data_metadata), transform_fn)
          | tft_beam.TransformDataset())
      
      transformed_data, transformed_metadata = transformed_dataset
      
      _ = (
          transformed_data
          | WriteToText('gs://BUCKET-NAME/output'))


if __name__ == '__main__':
  main()
