import os
import numpy as np
import tensorflow as tf
from data_reader import data_reader
from utils import ops
from utils.attention import multihead_attention_3d

"""
This module builds a standard U-NET for semantic segmentation.
"""


class PixelDCN(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        self.data_format = 'NHWC'
        self.initializer = tf.contrib.layers.xavier_initializer()
        if self.conf.data_type == '3D':
            self.conv_size = (3, 3, 3)
            self.pool_size = (2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [None, 160, 160, 8, 1]
            self.output_shape = [None, 160, 160, 8]
            self.initial_shape = [1, 160, 160, 8, 1]
        else:
            self.conv_size = (3, 3)
            self.pool_size = (2, 2)
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [
                self.conf.batch, self.conf.height, self.conf.width,
                self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.height, self.conf.width]

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.labels = tf.placeholder(
            tf.int64, self.output_shape, name='labels')
        self.global_uw = tf.get_variable("randomly_initialized_input", shape=self.initial_shape,initializer=self.initializer)
        self.coefficients = tf.placeholder(
            tf.float32, [3], name= 'coefficients' )
        self.predictions = self.inference(self.inputs)
        self.cal_loss()


    def get_loss(self, logits, label):
        #coefficients= tf.constant([3.0, 10.0, 1.0]) #$14899200
        num_classes = 3
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        label_flat = tf.reshape(label, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.coefficients), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def cal_loss(self):
        self.loss_op = self.get_loss(self.predictions, self.labels)
        self.probability = tf.nn.softmax(self.predictions)
        self.decoded_preds = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.labels, self.decoded_preds,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        _, self.acc_cls = tf.metrics.mean_per_class_accuracy(self.labels, self.decoded_preds, 3)


    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.scalar(name+'/neuron_accuracy', self.acc_cls[1]))
        summarys.append(tf.summary.scalar(name+'/back_accuracy', self.acc_cls[0]))
        if name == 'valid' and self.conf.data_type == '2D':
            summarys.append(
                tf.summary.image(name+'/input', self.inputs, max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/annotation',
                    tf.cast(tf.expand_dims(self.labels, -1),
                            tf.float32), max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/prediction',
                    tf.cast(tf.expand_dims(self.decoded_preds, -1),
                            tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        print('The input shape is', inputs.get_shape())
        outputs = ops.Conv3D(inputs=inputs,filters=self.conf.start_channel_num,kernel_size=3,strides=1)
        print('After intial conv, the shape is ', outputs.get_shape())

        down_outputs = []

        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.build_down_block(
                outputs, name, down_outputs, is_first, self.conf.isTraining)
            down_outputs.append(outputs)

        # go to bottom block
        outputs = self.build_bottom_block(outputs, 'bottom')


        uw_skips = []
        uw_out = ops.Conv3D(inputs=self.global_uw,filters=self.conf.start_channel_num,kernel_size=1,strides=1)
        uw_out = ops.BN_ReLU(uw_out, self.conf.isTraining)
     #   print('the shape of global Uw is', uw_out.get_shape())
        out_chan_num =self.conf.start_channel_num
        for layer_index in range(self.conf.network_depth-1):
            uw_out = ops.Conv3D(inputs=uw_out,filters=out_chan_num,kernel_size=3,strides=2)
            uw_out = ops.BN_ReLU(uw_out, self.conf.isTraining)
      #      print('the shape of global Uw is', uw_out.get_shape())
            uw_skips.append(uw_out)
            out_chan_num = out_chan_num*2
        

        # go to up block

        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            uw_input = uw_skips[layer_index]
            outputs = self.build_up_block(
                outputs, down_inputs, uw_input, name, self.conf.isTraining, is_final)

        ## go to out block 
        outputs = ops.Conv3D(inputs=outputs, filters=self.conf.class_num,kernel_size=1,	strides=1,use_bias=True)

        print('after a output conv, the shape is ', outputs.get_shape())

        return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False, training=True):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value

        inputs = ops.BN_ReLU(inputs, training)

        shortcut = ops.Conv3D(inputs=inputs,filters=out_num,kernel_size=1,strides=2)
        
        inputs = ops.Conv3D(inputs=inputs,filters=out_num,kernel_size=3,strides=2)
        inputs = ops.BN_ReLU(inputs, training)
        inputs = ops.Conv3D(inputs=inputs,filters=out_num,kernel_size=3,strides=1)
        # inputs = ops.BN_ReLU(inputs, training)
        # inputs = ops.Conv3D(inputs=inputs,filters=out_num,kernel_size=3,strides=1)
        out = inputs+shortcut
        print('after a down block, the shpe is', out.get_shape())
        return out

    def build_bottom_block(self, inputs, name):
        inputs = ops.BN_ReLU(inputs, self.conf.isTraining)

        out_num = inputs.shape[self.channel_axis].value
        
        inputs = ops.Conv3D(inputs=inputs,filters=out_num,kernel_size=3,strides=1)
    # out = multihead_attention_3d(
		# 			inputs, out_num, out_num, out_num, 2, self.conf.isTraining, layer_type='SAME')
        print('after a bottom block, the shape is ', inputs.get_shape())
        return inputs


    def reshape_range(self,tensor, i, j, shape):
        """Reshapes a tensor between dimensions i and j."""

        target_shape = tf.concat(
                [tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
                axis=0)

        return tf.reshape(tensor, target_shape)


    def flatten_3d(self, x):
        """flatten x."""

        x_shape = tf.shape(x)
        # [batch, heads, length, channels], length = d*h*w
        x = self.reshape_range(x, 1, 4, [tf.reduce_prod(x_shape[1:4])])

        return x


    def scatter_3d(self, x, shape):
        """scatter x."""

        x = tf.reshape(x, shape)

        return x


    def global_attention(self, down_inputs, uw_input, training):
        shape = [self.conf.batch,down_inputs.shape[1].value,down_inputs.shape[2].value, down_inputs.shape[3].value,down_inputs.shape[4].value]
        print('================shape',shape)
        print('the shape of downinput is', down_inputs.get_shape())
        print('the shape of uw_input is', uw_input.get_shape())
     #   down_input = ops.Conv3D(inputs=down_inputs,filters=down_inputs.shape[self.channel_axis].value,kernel_size=1,strides=1)
        down_input = self.flatten_3d(down_inputs)
    #    uw_input = ops.Conv3D(inputs=uw_input,filters=uw_input.shape[self.channel_axis].value,kernel_size=1,strides=1)
        uw_input = self.flatten_3d(uw_input)
        print('after reshaping, the shape of downinput is', down_input.get_shape())
        print('after reshaping, the shape of uw_input is', uw_input.get_shape())
        uw_input = tf.tile(uw_input, [self.conf.batch, 1,1])
        print('after tile, the shape of uw is', uw_input.get_shape())
        


        prob_map = tf.matmul(down_input,uw_input, transpose_b=True)
        print('the shape of prob map is ', prob_map.get_shape())
        prob_map = tf.nn.softmax(prob_map, axis=1)
        out_put = tf.matmul(prob_map, down_input, transpose_a=True)
        print('after multiply, the shape is ', out_put.get_shape())
        out_put = self.scatter_3d(out_put, shape)
        print('after scatter, the shape is', out_put.get_shape())
     #   out_put = ops.Conv3D(inputs=out_put,filters=out_put.shape[self.channel_axis].value,kernel_size=1,strides=1)
      #  out_put = ops.BN_ReLU(out_put, training)
        return out_put






    def attention_block(self, inputs, filters, training, projection_shortcut, strides, name):
        
        inputs = ops.BN_ReLU(inputs, training)
        shortcut = self.deconv_func()(
            inputs, filters, 3, name+'/deconv1',
            self.conf.data_type)
		# The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        if strides != 1:
            layer_type = 'UP'
        else:
            layer_type = 'SAME'
        
        inputs = multihead_attention_3d(inputs, filters, filters, filters, 1, training, layer_type)
        return inputs + shortcut        


    def build_up_block_attn(self, inputs, down_inputs, uw_input, name, training, final=False):
        out_num = inputs.shape[self.channel_axis].value
        down_inputs = self.global_attention(down_inputs, uw_input, training)

        inputs = tf.add(inputs, down_inputs, name=name+'/add')
        out = self.attention_block(inputs, out_num, self.conf.isTraining, 2, name)

        print('after an attn-based up block, the shape is ', out.get_shape())
        out = ops.BN_ReLU(inputs, self.conf.isTraining)
        return out        



    def build_up_block(self, inputs, down_inputs, uw_input, name, training,  final=False):
        out_num = inputs.shape[self.channel_axis].value
        down_inputs = self.global_attention(down_inputs, uw_input, training)
        inputs = tf.add(inputs, down_inputs, name=name+'/add')
        inputs = self.deconv_func()(
            inputs, out_num, 3, name+'/deconv1',
            self.conf.data_type)
        out_num = out_num/2
        conv2 = self.conv_func()(
            inputs, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        #out_num = out_num/2
        # conv3 = ops.conv(
        #     conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type)
        # print('after a regular up block, the shape is', conv3.get_shape())
        return conv2

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)

        train_reader = data_reader()

        for epoch_num in range(self.conf.max_step+1):
            if epoch_num and epoch_num % self.conf.test_interval == 0:
                inputs, labels, count = train_reader.get_random_test()
                inputs = np.expand_dims(inputs, axis=-1)
                feed_dict = {self.inputs: inputs, self.coefficients:  count,
                             self.labels: labels}
                loss, summary, acc_test, acc_all_test, _ = self.sess.run(
                    [self.loss_op, self.valid_summary,self.acc_cls,  self.accuracy_op, self.train_op], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                print('----testing loss', loss, '-------test acc neuron ', acc_test[1], '-------test acc back ', acc_test[0], '--------overall acc', acc_all_test)
            if epoch_num and epoch_num % self.conf.summary_interval == 0:
                inputs, labels, count = train_reader.next_batch(self.conf.batch)
                inputs = np.expand_dims(inputs, axis=-1)
                feed_dict = {self.inputs: inputs, self.coefficients:  count,
                             self.labels: labels}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, labels, count = train_reader.next_batch(self.conf.batch)
                inputs = np.expand_dims(inputs, axis=-1)
                feed_dict = {self.inputs: inputs, self.coefficients:  count,
                             self.labels: labels}
                loss, _ , acc, acc_all= self.sess.run(
                    [self.loss_op, self.train_op,self.acc_cls,  self.accuracy_op], feed_dict=feed_dict)
                print('----training loss', loss, '-------------training acc', acc[1], '-------test acc back ', acc[0], '-------overall acc', acc_all)
            if epoch_num and epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num+self.conf.reload_step)

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        if self.conf.data_type == '2D':
            test_reader = H5DataLoader(
                self.conf.data_dir+self.conf.test_data, False)
        else:
            test_reader = H53DDataLoader(
                self.conf.data_dir+self.conf.test_data, self.input_shape)
        self.sess.run(tf.local_variables_initializer())
        count = 0
        losses = []
        accuracies = []
        m_ious = []
        while True:
            inputs, labels = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.labels: labels}
            loss, accuracy, m_iou, _ = self.sess.run(
                [self.loss_op, self.accuracy_op, self.m_iou, self.miou_op],
                feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
            count += 1
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))
        print('M_iou: ', m_ious[-1])

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        data = data_reader()
        path = self.conf.sampledir + '/' +str(self.conf.test_step)
        if not os.path.exists(path): 
            os.makedirs(path)        
        overlap_size = [40,40,6]
        # for i in range(5):
        #     x_test, y_test, name, a,b, c= data.get_next_test()
        #     print('=========================')

        for test_id in range(0,data.test_size): #data.test_size
            print("Now reading patches from the image")
            x_test_whole, y_test_whole, name, patch_ids, shape= data.get_patches(test_id, overlap_size)
            print('the shape is ', shape)
            x_test_whole = np.expand_dims(x_test_whole, axis=-1)
            print ('Number of patches:', len(patch_ids))
            patch_num = len(patch_ids)
            overall_acc = 0
            prob_map = {}
            for i in range(patch_num):
            #   print('==================',i )
                x_test, y_test = data.get_patches_from_img(x_test_whole, y_test_whole, patch_ids[i])
                feed_dict2= {self.inputs:x_test, self.labels: y_test}
                acc, prob= self.sess.run([self.accuracy_op, self.probability], feed_dict= feed_dict2)
                overall_acc = overall_acc+acc
                prob = prob[0]
                location = patch_ids[i]

                for j in range(self.conf.height):
                    for k in range(self.conf.width):
                        for l in range(self.conf.depth):
                            key = (location[0]+j, location[1]+k, location[2]+l)
                            if key not in prob_map.keys():
                                prob_map[key] = []
                            prob_map[key].append(prob[j, k, l, :])                

            results = np.zeros((shape[0], shape[1], shape[2], self.conf.class_num),
                    dtype=np.float32)
            results2 = np.zeros((shape[0], shape[1], shape[2], self.conf.class_num),
                    dtype=np.float32)            

            for key in prob_map.keys():
                results[key[0],	key[1], key[2]] = np.mean(prob_map[key], axis=0)    
                results2[key[0],	key[1], key[2]] = np.max(prob_map[key], axis=0)           
            
            sample_path = path+'/'+name+'average.npz'
            sample_path2 = path+'/'+name+'max.npz'
            np.save(sample_path, results) 
            np.save(sample_path2, results2) 
            print('For the checkpoint', self.conf.test_step,' the test acc is', overall_acc) 




    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)


