from tensorflow.keras import backend as K

def _recall(y_true, y_pred):

    # 실제값과 예측값 모두 1이면 true positive, 그 개수
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    # 실제값이 1인 개수
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # reacall
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def _precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    #예측값이 1인 개수
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    #precision
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def _f1(y_true, y_pred):
    precision = _precision(y_true, y_pred)
    recall = _recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

