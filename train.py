# -*- coding: utf-8 -*-
# /usr/bin/env/python3
from nets.ann import inference
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from math import ceil as ceil
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
import time
import os

slim = tf.contrib.slim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=12, help='epoch to train the network')
    parser.add_argument('--number_of_bands', default=392, help='the input bands number')
    parser.add_argument('--num_output', default=12, help='the train images number')
    parser.add_argument('--prelogits_nodes', default=50,help='Nodes in last hidden layer')
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[4, 7, 9, 11])

    parser.add_argument('--dataset_path', default='content/gdrive/My Drive/IGARSSExperimentsData/train_without_outliers.csv', 
                            help='file to load for training')
    parser.add_argument('--test_fraction', default=0.2, help='fraction_used for training')
    parser.add_argument('--train_batch_size', default=32, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    
    parser.add_argument('--summary_path', default='summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep ckpt files')
    #parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=10, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=10, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=10, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=5, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='', help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)

    args = parser.parse_args()
    return args

def _add_loss_summaries(total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    """
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        summaries.append(tf.summary.scalar(l.op.name + ' (raw)', l))
        summaries.append(tf.summary.scalar(l.op.name, loss_averages.average(l)))
    """
    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op] + update_ops):
        train_op = tf.no_op(name='train')

    return train_op

def _preprocess_input(all_bands, label):

    selected_bands = []

    input_bands = np.zeros((392), np.float32)

    for band in selected_bands:
        input_bands[band] = all_bands[band]

    return input_bands, label
    

def batch_dataset(bands, labels):

    parse_func = lambda b, l: tuple(tf.py_func(_preprocess_input, [b, l], [tf.float32, label.float32]))

    dataset = tf.data.Dataset.from_tensor_slices((bands, labels))
    dataset = dataset.shuffle()
    dataset = dataset.map(parse_func)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()
    data_initializer = iterator.initializer
    next_element = iterator.get_next()

    return data_initializer, next_element

def load_dataset(dataset_path, test_fraction=0.2):
    cols = list([str(i) for i in range(392)])
    cols.append('label')

    dataset = pd.read_csv(dataset_path,names=cols)
    data_array = np.asarray(dataset.drop(['label'], axis=1))
    labels_array = np.array(dataset['label'])

    train_dataset = []
    train_labels = []
    test_dataset = []
    test_labels = []

    unique_classes = np.unique(labels_array)
    total_unique_classes = len(unique_classes)

    for label in unique_classes:
        all_ind = np.where(labels_array == label)[0]

        total_train =ceil( len(all_ind) * (1-test_fraction) )

        train_dataset.extend(data_array[all_ind[:total_train]])
        train_labels.extend(train_labels[all_ind[:total_train]])

        test_dataset.extend(test_dataset[all_ind[total_train:]])
        test_labels.extend(test_labels[all_ind[total_train:]])

    return np.array(train_dataset), np.array(train_labels), np.array(test_dataset), np.array(test_labels), total_unique_classes


if __name__ == '__main__':
    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args = get_parser()

        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # create ckpt path
        ckpt_dir = os.path.join(log_dir, args.ckpt_path)
        if not os.path.isdir(ckpt_dir):  # Create the log directory if it doesn't exist
            os.makedirs(ckpt_dir)

        # best create ckpt path
        ckpt_best_dir = os.path.join(log_dir, args.ckpt_best_path)
        if not os.path.isdir(ckpt_best_path):  # Create the log directory if it doesn't exist
            os.makedirs(ckpt_best_dir)


        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)

        # define placeholder
        inputs = tf.placeholder(name='bands', shape=[None, args.number_of_bands], dtype=tf.float32)
        labels = tf.placeholder(name='labels', shape=[None, 1], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')

        # load dataset
        train_bands,train_labels,test_bands,test_labels, total_classes = load_dataset(args.dataset_path, args.test_fraction)

        # prepare train dataset
        train_data_initializer, train_next_element = batch_dataset(train_bands, train_labels)

        # prepare test dataset
        test_data_initializer, test_next_element = batch_dataset(test_bands, test_labels)
        
        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        prelogits, net_points = inference(inputs, bottleneck_layer_size=args.prelogits_nodes, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)

        logits = slim.fully_connected(prelogits, total_classes, 
                weights_initializer=slim.initializers.xavier_initializer(), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

        # record the network architecture
        hd = open(os.path.join(log_dir,"arch.txt"), 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        # Norm for the prelogits
        eps = 1e-5
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # inference_loss, logit = cos_loss(prelogits, labels, args.num_output)
        w_init_method = slim.initializers.xavier_initializer()
        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        inference_loss = tf.reduce_sum(cross_entropy, name='inference_loss')
        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule, values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                         name='lr_schedule')
        
        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(os.path.join(log_dir, args.summary_path), sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), args.log_histograms)
        inc_epoch_op = tf.assign_add(inc_epoch_op, 1, name='inc_epoch_op')

        # record trainable variable
        hd = open(os.path.join(log_dir,"trainable_vars.txt"), 'w')
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        # init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        count = 0
        total_accuracy = {}
        pre_sec = 0
        for i in range(args.max_epoch):
            sess.run(train_data_initializer)
            _ = sess.run(inc_epoch_op)
            while True:
                try:
                    bands_for_train, labels_for_train = sess.run(train_next_element)

                    feed_dict = {inputs: bands_for_train, labels: labels_for_train, phase_train_placeholder: True}
                    start = time.time()
                    _, total_loss_val, inference_loss_val, reg_loss_val, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, regularization_losses, Accuracy_Op],
                             feed_dict=feed_dict)
                    end = time.time()

                    if (end-start) > 0:
                        pre_sec = args.train_batch_size/(end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                    # save summary
                    if count > 0 and count % args.summary_interval == 0:
                        feed_dict = {inputs: bands_for_train, labels: labels_for_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % args.ckpt_interval == 0:
                        print("\nSaving ckpt [%d]\n" % (count))
                        checkpoint_path = os.path.join(ckpt_dir, 'model-%s.ckpt' % subdir)
                        saver.save(sess, checkpoint_path, global_step=count, write_meta_graph=False)

                        metagraph_filename = os.path.join(ckpt_dir, 'model-%s.meta' % subdir)
                        if not os.path.exists(metagraph_filename):
                            saver.export_meta_graph(metagraph_filename)

                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')
                        nrof_batches = ceil(len(test_labels) / args.test_batch_size)
                        loss_array = np.zeros((nrof_batches,), np.float32)
                        xent_array = np.zeros((nrof_batches,), np.float32)
                        accuracy_array = np.zeros((nrof_batches,), np.float32)
                        v=0
                        sess.run(test_data_initializer)

                        while True:
                        try:
                            bands_for_test, labels_for_test = sess.run(test_next_element)
                            feed_dict = {inputs: bands_for_test, labels: labels_for_test, phase_train_placeholder: False}
                            total_loss_val, inference_loss_val, acc_val = \
                            sess.run([total_loss, inference_loss, Accuracy_Op],
                                    feed_dict=feed_dict)

                            loss_array[v], xent_array[v], accuracy_array[v] = (total_loss_val, inference_loss_val, acc_val)

                        except tf.errors.OutOfRangeError:
                            print('\nValidation Epoch: %d\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f\n' %
                                (count, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))
                            break

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break