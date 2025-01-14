import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

## specify the batch size
batch_size = 8

## specify the path where model is saved
model_path = os.path.abspath("../load_model/saved_model") 

date = '20231116'
input_dir = os.path.join('data', 'ecg', date)
output_root = os.path.join('data', 'ecg_feature', date)

participant_list = [i for i in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, i))]

for participant in participant_list:
    output_dir = os.path.join(output_root, participant)
    os.makedirs(output_dir, exist_ok=True)
    
    fn_list = [i for i in os.listdir(os.path.join(input_dir, participant)) if i.split('.')[-1] == 'npy']
    
    for fn in fn_list:
        input_path = os.path.join(input_dir, participant, fn)
        output_fn = fn
        
        x_ecg = np.load(input_path, allow_pickle=True)

        with tf.compat.v1.Session() as sess:
            saver       = tf.compat.v1.train.import_meta_graph(model_path + "/SSL_model.ckpt.meta")
            new_saver   = saver.restore(sess, tf.train.latest_checkpoint(model_path))

            graph       = tf.compat.v1.get_default_graph()
                
            input_tensor            = graph.get_tensor_by_name("input:0")
            drop_out                = graph.get_tensor_by_name("Drop_out:0")
            isTrain                 = graph.get_tensor_by_name(name = 'isTrain:0')

            conv1              = graph.get_tensor_by_name(name = 'conv_layer_1/kernel:0') #output after conv layer 1
            conv2              = graph.get_tensor_by_name(name = 'conv_layer_2/kernel:0') #output after conv layer 2
            flat_layer1        = graph.get_tensor_by_name(name = 'flat_layer1/Reshape:0') #output after conv layer 1
            conv3              = graph.get_tensor_by_name(name = 'conv_layer_3/kernel:0') #output after conv layer 3
            conv4              = graph.get_tensor_by_name(name = 'conv_layer_4/kernel:0') #output after conv layer 4
            flat_layer2        = graph.get_tensor_by_name(name = 'flat_layer2/Reshape:0') #output after conv layer 2
            conv5              = graph.get_tensor_by_name(name = 'conv_layer_5/kernel:0') #output after conv layer 5
            conv6              = graph.get_tensor_by_name(name = 'conv_layer_6/kernel:0') #output after conv layer 6
            flat_layer3        = graph.get_tensor_by_name(name = 'flat_layer3/Reshape:0') #output after conv block 3

            main_branch        = graph.get_tensor_by_name(name = 'flat_layer/Reshape:0') #output after all conv layers
            
            print('model loaded')
            
            print('Feature Shape: ', x_ecg.shape)

            length = np.shape(x_ecg)[0]    # calculate the length of sample ecg file
            feature_set = np.zeros((1, main_branch.get_shape()[1].value), dtype = int) # initialize an array to save extracted features
            steps = length //batch_size +1  
            for j in tqdm(range(steps)):
                signal_batch = x_ecg[np.mod(np.arange(j*batch_size,(j+1)*batch_size), length)] # get batch
                signal_batch = np.expand_dims(signal_batch, 2) # reshape to feed into 1D conv layers]
                print('Signal Batch Shape: ', signal_batch.shape)

                fetches = [main_branch] # fetching output from last conv layer
                fetched = sess.run(fetches, {input_tensor: signal_batch, isTrain: False, drop_out: 0.0})
                stacked = fetched[0]
                feature_set = np.vstack((feature_set, stacked)) # stacking extracted features
                
            x_ecg_feature = feature_set[1:length+1] # removing the first row
            
            print("feature_shape: ", x_ecg_feature.shape)
            np.save(os.path.join(output_dir, output_fn), x_ecg_feature)
            """
            x_ecg_feature can be further used to perform downstream task.
            """
    
