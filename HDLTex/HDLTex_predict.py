import os
from keras.callbacks import TensorBoard
from keras.utils import plot_model


os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"

import numpy as np
from keras.models import Sequential
import Data_helper
import BuildModel

if __name__ == "__main__":

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    MEMORY_MB_MAX = 1600000  # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 100  # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 50000  # Maximum number of unique words
    EMBEDDING_DIM = 300  # embedding dimension you can change it to {25, 100, 150, and 300} but need to change glove version
    batch_size_L1 = 64  # batch size in Level 1
    batch_size_L2 = 64  # batch size in Level 2
    epochs = 10

    L1_model = 1  # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
    L2_model = 2  # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2

    np.set_printoptions(threshold=np.inf)
    '''
    location of input data in two ways 
    1: Tokenizer that is using GLOVE
    2: loadData that is using couting words or tf-idf
    '''

    X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2, word_index, embeddings_index, number_of_classes_L1 = Data_helper.loadData_Tokenizer(
        MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, embedding_type='word2vec', is_train=False)
    number_of_classes_L1 = 6
    number_of_classes_L2 = [3, 5, 19, 3, 6, 12]
    word_index_length = 22796
    print('X_train:', len(X_train), 'y_train:', len(y_train), 'X_test:', len(X_test), 'y_test:', len(y_test),
          'content_L2_Train:', len(content_L2_Train), 'L2_Train:', len(L2_Train), 'content_L2_Test:',
          len(content_L2_Test), 'L2_Test:', len(L2_Test), 'number_of_classes_L2:', number_of_classes_L2, 'word_index:',
          len(word_index), 'number_of_classes_L1:', number_of_classes_L1)

    print("Loading Data is Done")

    #######################CNN Level 1########################
    if L1_model == 1:
        print('Create model of CNN')
        model = BuildModel.buildModel_CNN(word_index, embeddings_index, number_of_classes_L1, MAX_SEQUENCE_LENGTH,
                                          EMBEDDING_DIM, 1, word_index_length=word_index_length)
        model.load_weights('model/model.h5')
        preds = model.predict(X_test, batch_size=128)
        fout = open('output/preds.txt', 'w')
        for i in range(len(X_test)):
            fout.write(str(np.argsort(preds[i])[-1]) + '\n')

    #######################RNN Level 1########################
    if L1_model == 2:
        print('Create model of RNN')
        model = BuildModel.buildModel_RNN(word_index, embeddings_index, number_of_classes_L1, MAX_SEQUENCE_LENGTH,
                                          EMBEDDING_DIM, word_index_length=word_index_length)
        model.load_weights('model/model.h5')
        preds = model.predict(X_test, batch_size=128)
        fout = open('output/preds.txt', 'w')
        for i in range(len(X_test)):
            fout.write(str(np.argsort(preds[i])[-1]) + '\n')

    HDLTex = []  # Level 2 models is list of Deep Structure

    ######################CNN Level 2################################
    if L2_model == 1:
        for i in range(0, number_of_classes_L1):
            print('Create Sub model of CNN', i)
            HDLTex.append(Sequential())
            HDLTex[i] = BuildModel.buildModel_CNN(word_index, embeddings_index, number_of_classes_L2[i],
                                                  MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1, word_index_length=word_index_length)
            HDLTex[i].load_weights('model/model_' + str(i) + '.h5')
            preds = HDLTex[i].predict(content_L2_Test[i], batch_size=128)
            fout = open('output/preds_' + str(i) + '.txt', 'w')
            for j in range(len(content_L2_Test[i])):
                fout.write(str(np.argsort(preds[i])[-1]) + '\n')

    ######################RNN Level 2################################
    if L2_model == 2:
        for i in range(0, number_of_classes_L1):
            print('Create Sub model of RNN', i)
            HDLTex.append(Sequential())
            HDLTex[i] = BuildModel.buildModel_RNN(word_index, embeddings_index, number_of_classes_L2[i],
                                                  MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, word_index_length=word_index_length)
            HDLTex[i].load_weights('model/model_' + str(i) + '.h5')
            preds = HDLTex[i].predict(content_L2_Test[i], batch_size=128)
            fout = open('output/preds_' + str(i) + '.txt', 'w')
            for j in range(len(content_L2_Test[i])):
                fout.write(str(np.argsort(preds[i])[-1]) + '\n')
