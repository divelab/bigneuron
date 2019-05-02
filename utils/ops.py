import tensorflow as tf



"""
This module provides some short functions to reduce code volume
"""
def conv(inputs, out_num, kernel_size, scope, data_type='2D', norm=True):
    if data_type == '2D':
        outs = tf.layers.conv2d(
            inputs, out_num, kernel_size, padding='same', name=scope+'/conv',
            kernel_initializer=tf.truncated_normal_initializer)
    else:
        shape = list(kernel_size) + [inputs.shape[-1].value, out_num]
        weights = tf.get_variable(
            scope+'/conv/weights', shape,
            initializer=tf.truncated_normal_initializer())
        outs = tf.nn.conv3d(
            inputs, weights, (1, 1, 1, 1, 1), padding='SAME',
            name=scope+'/conv')
    if norm:
        return tf.contrib.layers.batch_norm(
            outs, decay=0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
            updates_collections=None, scope=scope+'/batch_norm')
    else:
        return tf.contrib.layers.batch_norm(
            outs, decay=0.9, epsilon=1e-5, activation_fn=None,
            updates_collections=None, scope=scope+'/batch_norm')



            
def Conv3D(inputs, filters, kernel_size, strides, use_bias=False):
	"""Performs 3D convolution without bias and activation function."""

	return tf.layers.conv3d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding='same',
			use_bias=use_bias,
			kernel_initializer=tf.truncated_normal_initializer())

def BN_ReLU(inputs, training):
	"""Performs a batch normalization followed by a ReLU6."""

	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide#common_fused_ops
	inputs = tf.layers.batch_normalization(
				inputs=inputs,
				axis=-1,
				momentum=0.997,
				epsilon=1e-5,
				center=True,
				scale=True,
				training=training, 
				fused=True)

	return tf.nn.relu6(inputs)

def deconv(inputs, out_num, kernel_size, scope, data_type, **kws):
  #  print(input_x)
 
    outputs = tf.layers.conv3d_transpose(inputs= inputs, filters= out_num, kernel_size= kernel_size, strides= 2, 
            padding = "SAME", activation =None, use_bias= False , name=scope + "deconv_layer")
    outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.99, center=True, activation_fn=None,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm', is_training= True)

    outputs = tf.nn.relu(outputs)
    
    return outputs



def pool(inputs, kernel_size, scope, data_type='2D'):
    if data_type == '2D':
        return tf.layers.max_pooling2d(inputs, kernel_size, (2, 2), name=scope)
    return tf.layers.max_pooling3d(inputs, kernel_size, (2, 2, 2), name=scope)