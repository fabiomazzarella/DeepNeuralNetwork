"""
This class implements a Deep Neural Network model:

Input:
1  - n_hidden_units = number of neurons for each hidden layer
2  - optimizer_class = numerical algorithm adopted within the weights optimization
3  - learning_rate = learning rate of the optimizer 
4  - batch_size = size of the mini-batch 
5  - activation = activation function for hidden layers 
6  - inizializer = weights initializers set to He initialisation
7  - manual_ini = set the kernel initialisation manually
8  - weights_ini = weights to be passed in order to initialise kernels
9  - batch_norm_momentum = momentum of the batch normalization
10 - dropout rate = if not None, a dropout regularization is adopted
11 - random_state
12 - l1_regularizer = if not None, a L1 regularization is applied with the passed penalty
13 - l2_regularizer = if not None, a L2 regularization is applied with the passed penalty
14 - clipping_th = gradient clipping threshold
15 - restore = restore a saved model 

"""

import logging, inspect
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import numpy as np
        
class DNN_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 n_hidden_units = [64, 64],
                 n_outputs = 2,
                 optimizer_class = tf.train.AdamOptimizer,
                 learning_rate = 0.01, 
                 batch_size = 20,
                 early_stopping_samples = 25,
                 activation = tf.nn.elu, 
                 initializer = tf.contrib.layers.variance_scaling_initializer(),
                 manual_ini = False,
                 weights_ini = None,
                 pert_sd = None,
                 batch_norm_momentum = None, 
                 dropout_rate = None, 
                 random_state = None,
                 l1_regularizer = 0,
                 l2_regularizer = 0,
                 clipping_th = 0,
                 restore = False):
        
        """Initialize the DNNClassifier"""
        self.n_hidden_units = np.array(n_hidden_units)
        #self.n_hidden_layers = self.n_hidden_units.shape[0]
        self.n_hidden_layers = sum(x > 0 for x in self.n_hidden_units)
        self.n_outputs = n_outputs
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.manual_ini = manual_ini
        self.weights_ini = weights_ini
        self.sd = pert_sd
        self.n_samples = early_stopping_samples
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.l1_regularizer = l1_regularizer
        self.l2_regularizer = l2_regularizer
        self.clipping_th = clipping_th
        self.restore = restore
        self._session = None

    # to make this notebook's output stable across runs
    def _reset_graph(self):
        tf.reset_default_graph()
        tf.set_random_seed(self.random_state)
        np.random.seed(self.random_state)
    
    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and dropout."""
        print(str(self.n_samples))
        print('The number of Hidden Layers is: ' + str(self.n_hidden_layers))
        with tf.name_scope('dnn'):
            for layer in range(self.n_hidden_layers):
                n_neurons = self.n_hidden_units[layer]
                print('Number of neurons of layer no. ' + str(layer+1) + ': ' + str(n_neurons))
                if n_neurons != 0:
                    if self.dropout_rate:
                        print('Drop-out')
                        inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
                    if self.manual_ini==True:
                        print('Manual Weights Initialisation...')
                        if self.weights_ini is not None:
                            W = self.weights_ini[layer] + np.random.normal(0, self.sd)
                            print(W)
                        else:
                            print('... with He Initialisation')
                            n_in = int(inputs.shape[1])
                            n_out = n_neurons
                            low_val = -np.sqrt(2) * np.sqrt(6.0 / (n_in + n_out)) 
                            high_val = np.sqrt(2) * np.sqrt(6.0 / (n_in + n_out)) 
                            W = np.random.uniform(low = low_val, high = high_val, size = [n_in, n_out])
                        inputs = tf.layers.dense(inputs, n_neurons,
                                                 kernel_initializer = tf.constant_initializer(W, dtype=tf.float32),
                                                 name = "hidden%d" % (layer + 1))
                    else:
                        print('Automatic Weights Initialisation')
                        inputs = tf.layers.dense(inputs, n_neurons,
                                                 kernel_initializer = self.initializer,
                                                 name = "hidden%d" % (layer + 1))
            
                    if self.batch_norm_momentum:
                        print('Batch Normalization')
                        inputs = tf.layers.batch_normalization(inputs, momentum = self.batch_norm_momentum,
                                                           training = self._training)
                    inputs = self.activation(inputs, name = "hidden%d_out" % (layer + 1))              
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        """Build the DNN model"""
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
            y = tf.placeholder(tf.int32, shape=(None), name="y")

        with tf.name_scope('batch_normalization'):
            if self.batch_norm_momentum or self.dropout_rate:
                self._training = tf.placeholder_with_default(False, shape=(), name='training')
            else:
                self._training = None
        
        dnn_outputs = self._dnn(X)
        if self.manual_ini==True:
            print('Manual Weights Initialisation')
            if self.weights_ini is not None:
                W = self.weights_ini[-1] + np.random.normal(0, self.sd)
                print(W)
            else:
                print('...with He Initialisation')
                n_in = int(dnn_outputs.shape[1])
                n_out = n_outputs
                low_val = -np.sqrt(2) * np.sqrt(6.0 / (n_in + n_out)) 
                high_val = np.sqrt(2) * np.sqrt(6.0 / (n_in + n_out)) 
                W = np.random.uniform(low = low_val, high = high_val, size = [n_in, n_out])
            logits = tf.layers.dense(dnn_outputs, n_outputs,
                                     kernel_initializer = tf.constant_initializer(W, dtype=tf.float32),
                                     name = "logits")
        else:
            print('Automatic Weights Initialisation')
            logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer = self.initializer, name = "logits")
        
        
        Y_proba = tf.nn.softmax(logits, name = "Y_proba")

        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
 
        #Retrieve the weights for every layer
        for layer in range(self.n_hidden_layers):
            globals()['w' + str(layer+1)] = tf.get_default_graph().get_tensor_by_name('hidden%d/kernel:0' % (layer + 1))
        globals()['w' + str(layer+2)] = tf.get_default_graph().get_tensor_by_name("logits/kernel:0")
        
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
        base_loss = tf.reduce_mean(xentropy, name='loss')
        
        if self.l1_regularizer!=0:
            #reg_losses = tf.get_variable('reg', shape=(), initializer=tf.zeros_initializer())
            reg_losses = 0
            for layer in range(self.n_hidden_layers + 1):
                w = globals()['w' + str(layer+1)]
                #reg_losses = tf.add(reg_losses, tf.reduce_sum(tf.abs(w)))
                reg_losses = reg_losses + tf.reduce_sum(tf.abs(w))
                #reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)) + tf.reduce_sum(tf.abs(W3))
            #loss = tf.add(base_loss, self.l1_regularizer * reg_losses, name='loss')    
            loss = tf.reduce_mean(base_loss + self.l1_regularizer * reg_losses, name='loss')
        elif self.l2_regularizer!=0:
            #reg_losses = tf.get_variable('reg', shape=(), initializer=tf.zeros_initializer())
            reg_losses = 0
            for layer in range(self.n_hidden_layers + 1):
                w = globals()['w' + str(layer+1)]
                reg_losses = reg_losses + tf.reduce_sum(tf.nn.l2_loss(w))
                #reg_losses = tf.add(reg_losses, tf.reduce_sum(tf.square(w)))
                #reg_losses = tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(W3))
            #loss = tf.add(base_loss, self.l2_regularizer * reg_losses, name='loss')
            loss = tf.reduce_mean(base_loss + self.l2_regularizer * reg_losses, name='loss')
        else:        
            loss = base_loss 
        
        
        
        
        with tf.name_scope('train'):
            optimizer = self.optimizer_class(learning_rate=self.learning_rate)
            #Possibility for gradient clipping
            if self.clipping_th !=0:
                threshold = self.clipping_th
                grads_and_vars = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
                training_op = optimizer.apply_gradients(capped_gvs)
            else: 
                training_op = optimizer.minimize(loss)

        #with tf.name_scope('evaluation'):
        correct = tf.nn.in_top_k(logits, y, 1, name = "Y_pred")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
         
        #file_writer = tf.summary.FileWriter('/home/innovation/notebooks/tf_logs', tf.get_default_graph())    
            
        #Create a summary to monitor loss tensor
        tf.summary.scalar('LOSS', loss)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar('ACCURACY', accuracy)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        
        
        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver
        self._merged_op = merged_summary_op
    
    def close_session(self):
        if self._session:
            self._session.close()

    
    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving to disk)"""
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    
    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""
        self.close_session()

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)
        
        # Translate the labels vector to a vector of sorted class indices, containing
        # integers from 0 to n_outputs - 1.
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class
        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)
        
        self._reset_graph()
        self._graph = tf.Graph()
        
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
        max_checks_without_progress = self.n_samples
        print('Checks without progress: ' + str(max_checks_without_progress))
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        
        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            train_writer = tf.summary.FileWriter('/home/innovation/notebooks/tf_logs' + '/train', sess.graph)
            test_writer  = tf.summary.FileWriter('/home/innovation/notebooks/tf_logs' + '/test')
            # op to write logs to Tensorboard
            #summary_writer = tf.summary.FileWriter('/home/innovation/notebooks/tf_logs', graph=tf.get_default_graph())
            #merged_summary_op = tf.summary.merge_all()
            if self.restore == True:
                saver = tf.train.import_meta_graph("results/model.ckpt.meta")
                saver.restore(sess, "results/model.ckpt")
                #saver.restore(sess,tf.train.latest_checkpoint('results/'))
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val, summary_val = sess.run([self._loss, self._accuracy, self._merged_op],
                                                               feed_dict={self._X: X_valid,
                                                                          self._y: y_valid})
             
                    # Write logs at every iteration
                    test_writer.add_summary(summary_val, epoch)
             
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > np.int(max_checks_without_progress):
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train, summary_train = sess.run([self._loss, self._accuracy, self._merged_op],
                                                     feed_dict={self._X: X_batch,
                                                                self._y: y_batch})
                    #accuracy_summary = tf.summary.scalar('Accuracy', acc_train)
                    train_writer.add_summary(summary_train, epoch)
                    #file_writer = tf.summary.FileWriter('/home/innovation/notebooks/tf_logs', tf.get_default_graph())
                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(
                        epoch, loss_train, acc_train * 100))
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            return self

    def pred_prob_pretrained(self, X):
        """Predict class probabilities using a pre-trained model"""    
        self.close_session()
        self._reset_graph()
        self._graph = tf.Graph()
        n_inputs = X.shape[1]
        n_outputs = self.n_outputs
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            saver = tf.train.import_meta_graph("results/model.ckpt.meta")
            saver.restore(sess, "results/model.ckpt")
            pred_prob = sess.run([self._Y_proba], feed_dict={self._X: X})
            return pred_prob
           
        
    def predict_proba(self, X):
        """Predict class probabilities for the trained model"""    
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        """Predict classes for the fitted model"""    
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        """Save the model in the specified path"""     
        self._saver.save(self._session, path)
        
    def loss(self, X, y):
         with self._session.as_default() as sess:
            return self._loss.eval(feed_dict={self._X: X,
                                              self._y: y})
        



