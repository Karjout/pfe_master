from tensorflow.keras import backend as K
from adabelief_tf import AdaBeliefOptimizer
import joblib
import tensorflow as tf


def load_dl_model():

    # evaluation function for ANN model
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # load ANN model
    nn_model = tf.keras.models.load_model("model/NN4-0590.hdf5",
                                          custom_objects={'AdaBeliefOptimizer': AdaBeliefOptimizer, 'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    return nn_model


# load scaler for DL model
def load_scaler_dl():
    min_max = joblib.load('feature_scaling_dl/min_max_scaler.pkl')
    return min_max
