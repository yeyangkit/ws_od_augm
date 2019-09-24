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
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from shutil import copyfile
import sys
import smtplib

from object_detection import model_hparams
from object_detection import model_lib

tf.logging.set_verbosity(tf.logging.INFO)

from absl import flags

flags.DEFINE_string(
    'model_parent_dir', '/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/model_dirs/AUGMENT_structure',
    'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'dirname', 'sharedEncoder', 'name prefix.')
flags.DEFINE_boolean(
    'reload_pretrained_model', False, 'reload_pretrained_model and ignore the model_parent_dir and dirname')

import datetime

now = datetime.datetime.now()
flags.DEFINE_string('time_stample', now.strftime("%Y_%m_%d_%H_%M_%S"), 'current time')

# flags.DEFINE_string(
#     'model_dir', "{}/{}".format(flags.FLAGS.model_parent_dir, flags.FLAGS.time_stample), 'Path to output model directory '
#     'where event and checkpoint files will be written.')

flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                                                  'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job. Note '
                     'that one call only use this in eval-only mode, and '
                     '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                                                       'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                                                                'one of every n train input examples for evaluation, '
                                                                'where n is provided. This is only used if '
                                                                '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
                               'represented as a string containing comma-separated '
                               'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
                            '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
                            'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
                       'one round of eval vs running continuously (default).'
)
flags.DEFINE_integer('save_checkpoints_steps', 3000, 'Evaluation after saving checkpoint.')
flags.DEFINE_integer('tf_random_seed', 1, 'Random seed for weight initialization.')

FLAGS = flags.FLAGS


# class EmailSender(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#
#     # email setup
#     port = 465  # For SSL
#     smtp_server = "smtp.gmail.com"
#     sender_email = "likunleo@gmail.com"  # Enter your address
#     receiver_email = "likunleo@gmail.com"  # Enter receiver address
#     password = ""
#     message = str(epoch) + " " + str(logs.get('loss')) # todo
#
#     context = ssl.create_default_context()
#     with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#         server.login(sender_email, password)
#         server.sendmail(sender_email, receiver_email, message)

def main(unused_argv):
    # sys.path.append('/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map')
    # sys.path.append('/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/slim')
    os.system(
        'export PYTHONPATH=/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map:/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/slim')
    print("=================    system path     =================\n" + str(
        sys.path) + "\n=================       ===================\n")
    # sleep(15000)
    # flags.mark_flag_as_required('model_parent_dir')
    # flags.mark_flag_as_required('model_dir')

    if not flags.FLAGS.reload_pretrained_model:
        flags.DEFINE_string(
            'model_dir', "{}/{}_{}".format(flags.FLAGS.model_parent_dir, flags.FLAGS.dirname, flags.FLAGS.time_stample),
            'Path to output model directory '
            'where event and checkpoint files will be written.')
    else:
        flags.DEFINE_string(
            'model_dir', None,
            'Path to output model directory '
            'where event and checkpoint files will be written.')

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    flags.mark_flag_as_required('pipeline_config_path')
    copyfile(FLAGS.pipeline_config_path, os.path.join(FLAGS.model_dir + '/correspondingPipelineConfig.config'))

    copyfile(
        '/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/object_detection/meta_architectures/ssd_augmentation_meta_arch.py',
        os.path.join(FLAGS.model_dir + '/ssd_augmentation_meta_arch.py'))

    copyfile(
        '/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/object_detection/predictors/u_net_predictor.py',
        os.path.join(FLAGS.model_dir + '/u_net_predictor.py'))
    copyfile(
        '/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/object_detection/predictors/upsampling_predictor.py',
        os.path.join(FLAGS.model_dir + '/upsampling_predictor.py'))
    copyfile(
        '/mrtstorage/users/students/yeyang/ws/ws_od_augm/tensorflow_grid_map/object_detection/predictors/ht_predictor.py',
        os.path.join(FLAGS.model_dir + '/ht_predictor.py'))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    session_config=sess_config,
                                    tf_random_seed=FLAGS.tf_random_seed,
                                    save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(input_fn,
                               steps=None,
                               checkpoint_path=tf.train.latest_checkpoint(
                                   FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                      train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

    email_sender = EmailSender()


if __name__ == '__main__':
    tf.app.run()
