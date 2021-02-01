import pandas as pd
import numpy as np
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow.compat.v1 as tf
from datetime import timedelta
from tqdm import tqdm

class gru2:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.GRUCell(size_layer)
        
        with tf.variable_scope('forward', reuse = False):
            rnn_cells_forward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)],
                state_is_tuple = False,
            )
            self.X_forward = tf.placeholder(tf.float32, (None, None, size))
            drop_forward = tf.nn.rnn_cell.DropoutWrapper(
                rnn_cells_forward, output_keep_prob = forget_bias
            )
            self.hidden_layer_forward = tf.placeholder(
                tf.float32, (None, num_layers * size_layer)
            )
            self.outputs_forward, self.last_state_forward = tf.nn.dynamic_rnn(
                drop_forward,
                self.X_forward,
                initial_state = self.hidden_layer_forward,
                dtype = tf.float32,
            )

        with tf.variable_scope('backward', reuse = False):
            rnn_cells_backward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)],
                state_is_tuple = False,
            )
            self.X_backward = tf.placeholder(tf.float32, (None, None, size))
            drop_backward = tf.nn.rnn_cell.DropoutWrapper(
                rnn_cells_backward, output_keep_prob = forget_bias
            )
            self.hidden_layer_backward = tf.placeholder(
                tf.float32, (None, num_layers * size_layer)
            )
            self.outputs_backward, self.last_state_backward = tf.nn.dynamic_rnn(
                drop_backward,
                self.X_backward,
                initial_state = self.hidden_layer_backward,
                dtype = tf.float32,
            )

        self.outputs = self.outputs_backward - self.outputs_forward
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        self.logits = tf.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
        
    def calculate_accuracy(self, real, predict):
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100
    
    def anchor(self, signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer
    
    def forecast(self,
                 modelnn, 
                 epoch, 
                 num_layers, 
                 size_layer, 
                 df, 
                 df_train, 
                 timestamp, 
                 test_size, 
                 minmax):        
        
        # tf.reset_default_graph()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    
        pbar = tqdm(range(epoch), desc = 'train loop')
        for i in pbar:
            init_value_forward = np.zeros((1, num_layers * size_layer))
            init_value_backward = np.zeros((1, num_layers * size_layer))
            total_loss, total_acc = [], []
            for k in range(0, df_train.shape[0] - 1, timestamp):
                index = min(k + timestamp, df_train.shape[0] - 1)
                batch_x_forward = np.expand_dims(
                    df_train.iloc[k : index, :].values, axis = 0
                )
                batch_x_backward = np.expand_dims(
                    np.flip(df_train.iloc[k : index, :].values, axis = 0), axis = 0
                )
                batch_y = df_train.iloc[k + 1 : index + 1, :].values
                logits, last_state_forward, last_state_backward, _, loss = sess.run(
                    [
                        modelnn.logits,
                        modelnn.last_state_forward,
                        modelnn.last_state_backward,
                        modelnn.optimizer,
                        modelnn.cost,
                    ],
                    feed_dict = {
                        modelnn.X_forward: batch_x_forward,
                        modelnn.X_backward: batch_x_backward,
                        modelnn.Y: batch_y,
                        modelnn.hidden_layer_forward: init_value_forward,
                        modelnn.hidden_layer_backward: init_value_backward,
                    },
                )
                init_value_forward = last_state_forward
                init_value_backward = last_state_backward
                total_loss.append(loss)
                total_acc.append(self.calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            pbar.set_postfix(cost = np.mean(total_loss), acc = np.mean(total_acc))
        
        future_day = test_size

        output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // timestamp) * timestamp
        init_value_forward = np.zeros((1, num_layers * size_layer))
        init_value_backward = np.zeros((1, num_layers * size_layer))
    
        for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
            batch_x_forward = np.expand_dims(
            df_train.iloc[k : k + timestamp, :], axis = 0
            )
            batch_x_backward = np.expand_dims(
                np.flip(df_train.iloc[k : k + timestamp, :].values, axis = 0), axis = 0
            )
            out_logits, last_state_forward, last_state_backward = sess.run(
                [
                    modelnn.logits,
                    modelnn.last_state_forward,
                    modelnn.last_state_backward,
                ],
                feed_dict = {
                    modelnn.X_forward: batch_x_forward,
                    modelnn.X_backward: batch_x_backward,
                    modelnn.hidden_layer_forward: init_value_forward,
                    modelnn.hidden_layer_backward: init_value_backward,
                },
            )
            init_value_forward = last_state_forward
            init_value_backward = last_state_backward
            output_predict[k + 1 : k + timestamp + 1, :] = out_logits
    
        if upper_b != df_train.shape[0]:
            batch_x_forward = np.expand_dims(df_train.iloc[upper_b:, :], axis = 0)
            batch_x_backward = np.expand_dims(
                np.flip(df_train.iloc[upper_b:, :].values, axis = 0), axis = 0
            )
            out_logits, last_state_forward, last_state_backward = sess.run(
                [modelnn.logits, modelnn.last_state_forward, modelnn.last_state_backward],
                feed_dict = {
                    modelnn.X_forward: batch_x_forward,
                    modelnn.X_backward: batch_x_backward,
                    modelnn.hidden_layer_forward: init_value_forward,
                    modelnn.hidden_layer_backward: init_value_backward,
                },
            )
            init_value_forward = last_state_forward
            init_value_backward = last_state_backward
            output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
            future_day -= 1
            date_ori.append(date_ori[-1] + timedelta(days = 1))
            
        init_value_forward = last_state_forward
        init_value_backward = last_state_backward
        
        for i in range(future_day):
            o = output_predict[-future_day - timestamp + i:-future_day + i]
            o_f = np.flip(o, axis = 0)
            out_logits, last_state_forward, last_state_backward = sess.run(
                [
                    modelnn.logits,
                    modelnn.last_state_forward,
                    modelnn.last_state_backward,
                ],
                feed_dict = {
                    modelnn.X_forward: np.expand_dims(o, axis = 0),
                    modelnn.X_backward: np.expand_dims(o_f, axis = 0),
                    modelnn.hidden_layer_forward: init_value_forward,
                    modelnn.hidden_layer_backward: init_value_backward,
                },
            )
            init_value_forward = last_state_forward
            init_value_backward = last_state_backward
            output_predict[-future_day + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days = 1))

        output_predict = minmax.inverse_transform(output_predict)
        deep_future = self.anchor(output_predict[:, 0], 0.3)
                
        return deep_future