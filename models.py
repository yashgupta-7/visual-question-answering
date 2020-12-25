from keras.losses import categorical_crossentropy
import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K

from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Embedding, Concatenate, Dot, Multiply, Bidirectional, InputLayer, LayerNormalization
from keras.optimizers import Adam
from keras.layers.core import Reshape, Activation, Dropout

##LSTM-CNN
def create_model1():
    """LSTM-CNN Model Implementation aka model1"""
  # with strategy.scope():
    MAX_LEN = 26
    VOCAB_SIZE = 12602
    num_hidden_layers_lstm = 2
    word_vec_dim = 300

    model_lang = Sequential()
    model_lang.add(InputLayer(input_shape=(MAX_LEN,)))  ## placeholder for input layer
    model_lang.add(Embedding(VOCAB_SIZE+1, word_vec_dim))              ## (batch_size, input_length) -> (batch_size, input_length, output_dim)
    model_lang.add(Activation('tanh'))
    model_lang.add(Dropout(0.5))

    x1_w, x1_h, x1_c = LSTM(units = 512, return_sequences=True, input_shape=(MAX_LEN, word_vec_dim), return_state=True)(model_lang.output)
    x2_w, x2_h, x2_c = LSTM(units = 512, return_sequences=True, return_state=True)(x1_w)
    x3_f = Concatenate(axis=1)([x1_h, x1_c, x2_h, x2_c])
    x3_f = Dense(1024)(x3_f)
    x3_f = Activation('tanh')(x3_f)
    x3_f = Dropout(0.5)(x3_f)
    # if num_hidden_layers_lstm == 1:
    #   model_lang.add(LSTM(units = 512, return_sequences=False, input_shape=(QUES_LEN, word_vec_dim))) ##  [batch, timesteps, feature] -> [batch_size, output_len]
    # else:
      # model_lang.add(LSTM(units = 512, return_sequences=True, input_shape=(QUES_LEN, word_vec_dim)))
      # for i in range(num_hidden_layers_lstm-2):
      #   model_lang.add(LSTM(units = 512, return_sequences=True))
      # model_lang.add(LSTM(units = 512, return_sequences=False))

    model_image = Sequential()
    model_image.add(Reshape((4096,), input_shape = (4096,)))
    model_image.add(LayerNormalization(axis=1))
    # model_image.add(Dense(2048))
    # model_image.add(Activation("tanh"))
    # model_image.add(Dropout(0.5))
    model_image.add(Dense(1024))
    model_image.add(Activation("tanh"))
    model_image.add(Dropout(0.5))

    x = Multiply()([x3_f, model_image.output]) ##Concatenate(axis=1)([x3_f, model_image.output])
    num_dense_layers = 1
    for _ in range(num_dense_layers):
        x = Dense(1024, kernel_initializer='uniform')(x)
        x = Activation("tanh")(x)
        x = Dropout(0.5)(x)
    
    x = Dense(1001)(x)
    x = Activation("softmax")(x)

    model = Model([model_image.input, model_lang.input], [x])

    return model


#LSTM-SPACY Embeddings
def create_model2():
  # with strategy.scope():
    MAX_LEN = 26
    VOCAB_SIZE = 12602
    num_hidden_layers_lstm = 2
    word_vec_dim = 300

    model_lang = Sequential()
    model_lang.add(InputLayer(input_shape=(MAX_LEN,word_vec_dim)))  ## placeholder for input layer
    # model_lang.add(Embedding(VOCAB_SIZE+1, word_vec_dim))              ## (batch_size, input_length) -> (batch_size, input_length, output_dim)
    # model_lang.add(Activation('tanh'))
    # model_lang.add(Dropout(0.5))

    x1_w, x1_h, x1_c = LSTM(units = 512, return_sequences=True, input_shape=(MAX_LEN, word_vec_dim), return_state=True)(model_lang.output)
    x2_w, x2_h, x2_c = LSTM(units = 512, return_sequences=True, return_state=True)(x1_w)
    x3_f = Concatenate(axis=1)([x1_h, x1_c, x2_h, x2_c])
    x3_f = Dense(1024)(x3_f)
    x3_f = Activation('tanh')(x3_f)
    x3_f = Dropout(0.5)(x3_f)
    # if num_hidden_layers_lstm == 1:
    #   model_lang.add(LSTM(units = 512, return_sequences=False, input_shape=(QUES_LEN, word_vec_dim))) ##  [batch, timesteps, feature] -> [batch_size, output_len]
    # else:
      # model_lang.add(LSTM(units = 512, return_sequences=True, input_shape=(QUES_LEN, word_vec_dim)))
      # for i in range(num_hidden_layers_lstm-2):
      #   model_lang.add(LSTM(units = 512, return_sequences=True))
      # model_lang.add(LSTM(units = 512, return_sequences=False))

    model_image = Sequential()
    model_image.add(Reshape((4096,), input_shape = (4096,)))
    model_image.add(LayerNormalization(axis=1))
    # model_image.add(Dense(2048))
    # model_image.add(Activation("tanh"))
    # model_image.add(Dropout(0.5))
    model_image.add(Dense(1024))
    model_image.add(Activation("tanh"))
    model_image.add(Dropout(0.5))

    x = Multiply()([x3_f, model_image.output]) ##Concatenate(axis=1)([x3_f, model_image.output])
    num_dense_layers = 1
    for _ in range(num_dense_layers):
        x = Dense(1024, kernel_initializer='uniform')(x)
        x = Activation("tanh")(x)
        x = Dropout(0.5)(x)
    
    x = Dense(1001)(x)
    x = Activation("softmax")(x)

    model = Model([model_image.input, model_lang.input], [x])

    return model


## BiLSTM_CNN
def create_model3():
    """BiLSTM-CNN Model Implementation aka model3"""
  # with strategy.scope():
    MAX_LEN = 26
    VOCAB_SIZE = 12602
    num_hidden_layers_lstm = 2
    word_vec_dim = 300
    drop = 0.5

    model_lang = Sequential()
    model_lang.add(InputLayer(input_shape=(MAX_LEN,)))  ## placeholder for input layer
    model_lang.add(Embedding(VOCAB_SIZE+1, word_vec_dim))              ## (batch_size, input_length) -> (batch_size, input_length, output_dim)
    model_lang.add(Activation('tanh'))
    model_lang.add(Dropout(drop))

    x1_w, x1_h, x1_c, x1_bh, x1_bc = Bidirectional(LSTM(units = 512, return_sequences=True, input_shape=(MAX_LEN, word_vec_dim), return_state=True))(model_lang.output)
    x2_w, x2_h, x2_c, x2_bh, x2_bc = Bidirectional(LSTM(units = 512, return_sequences=True, return_state=True))(x1_w)
    x3_f = Concatenate(axis=1)([x1_h, x1_c, x1_bh, x1_bc, x2_h, x2_c, x2_bh, x2_bc])
    x3_f = Dense(1024)(x3_f)
    x3_f = Activation('tanh')(x3_f)
    x3_f = Dropout(drop)(x3_f)
    # if num_hidden_layers_lstm == 1:
    #   model_lang.add(LSTM(units = 512, return_sequences=False, input_shape=(QUES_LEN, word_vec_dim))) ##  [batch, timesteps, feature] -> [batch_size, output_len]
    # else:
      # model_lang.add(LSTM(units = 512, return_sequences=True, input_shape=(QUES_LEN, word_vec_dim)))
      # for i in range(num_hidden_layers_lstm-2):
      #   model_lang.add(LSTM(units = 512, return_sequences=True))
      # model_lang.add(LSTM(units = 512, return_sequences=False))

    model_image = Sequential()
    model_image.add(Reshape((4096,), input_shape = (4096,)))
    model_image.add(LayerNormalization(axis=1))
    # model_image.add(Dense(2048))
    # model_image.add(Activation("tanh"))
    # model_image.add(Dropout(0.5))
    model_image.add(Dense(1024))
    model_image.add(Activation("tanh"))
    model_image.add(Dropout(drop))

    x = Multiply()([x3_f, model_image.output]) ##Concatenate(axis=1)([x3_f, model_image.output])
    num_dense_layers = 1
    for _ in range(num_dense_layers):
        x = Dense(1024, kernel_initializer='uniform')(x)
        x = Activation("tanh")(x)
        x = Dropout(drop)(x)
    
    x = Dense(1001)(x)
    x = Activation("softmax")(x)

    model = Model([model_image.input, model_lang.input], [x])

    return model

#BiLSTM-VIS
def create_model4():
  """Vis-BiLSTM Model Implementation aka model4"""
  MAX_LEN = 26
  VOCAB_SIZE = 12602
  num_hidden_layers_lstm = 2
  word_vec_dim = 300
  drop = 0.7

  model_lang = Sequential()
  model_lang.add(InputLayer(input_shape=(MAX_LEN,)))  ## placeholder for input layer
  model_lang.add(Embedding(VOCAB_SIZE+1, word_vec_dim))              ## (batch_size, input_length) -> (batch_size, input_length, output_dim)
  model_lang.add(Activation('tanh'))
  model_lang.add(Dropout(drop))

  model_image = Sequential()
  model_image.add(Reshape((4096,), input_shape = (4096,)))
  model_image.add(LayerNormalization(axis=1))
  model_image.add(Dense(300))
  model_image.add(Activation("tanh"))
  model_image.add(Dropout(drop))
  model_image.add(Reshape((1,300)))

  model_image2 = Sequential()
  model_image2.add(Reshape((4096,), input_shape = (4096,)))
  model_image2.add(LayerNormalization(axis=1))
  model_image2.add(Dense(300))
  model_image2.add(Activation("tanh"))
  model_image2.add(Dropout(drop))
  model_image2.add(Reshape((1,300)))

  x = Concatenate(axis=1)([model_lang.output, model_image.output, model_image2.output])

  # print(x.shape)
  x1_w, x1_h, x1_c, x1_bh, x1_bc = Bidirectional(LSTM(units = 512, return_sequences=True, input_shape=(MAX_LEN+2, word_vec_dim), return_state=True))(Dropout(drop)(x))
  x2_w, x2_h, x2_c, x2_bh, x2_bc = Bidirectional(LSTM(units = 512, return_sequences=True, return_state=True))(Dropout(drop)(x1_w))
  x3_f = Concatenate(axis=1)([x1_h, x1_c, x1_bh, x1_bc, x2_h, x2_c, x2_bh, x2_bc])
  x3_f = Dense(1024)(x3_f)
  x3_f = Activation('tanh')(x3_f)
  x3_f = Dropout(drop)(x3_f)

  x = Dense(1001)(x3_f)
  x = Activation("softmax")(x)

  main_model = Model([model_image.input, model_image2.input, model_lang.input], [x])
	
  return main_model