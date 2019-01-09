import tensorflow as tf
slim = tf.contrib.slim

def inference(inputs, bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005, reuse=False, keep_prob=0.5):

    end_points = {}
    
    with tf.variable_scope(scope, 'ANN', [inputs], reuse=reuse) as scope:

        # Hidden Layer 1
        net = slim.fully_connected(inputs, 300, 
                    weights_initializer=slim.initializers.xavier_initializer(), 
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    scope='hidden_1', reuse=False)

        end_points['hidden_1'] = net

        # Hidden Layer 2
        net = slim.fully_connected(net, 150, 
                    weights_initializer=slim.initializers.xavier_initializer(), 
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    scope='hidden_2', reuse=False)

        end_points['hidden_2'] = net

        # Hidden Layer 3
        net = slim.fully_connected(net, bottleneck_layer_size, 
                    weights_initializer=slim.initializers.xavier_initializer(), 
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    scope='hidden_3', reuse=False)

        end_points['hidden_3'] = net

        # Drop out
        net = def dropout(net,keep_prob=keep_prob)

        end_points['dropout'] = net

        return net, end_points


        
