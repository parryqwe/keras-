# keras-
## 套件
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras import losses
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback:
import keras.backend as K
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
## 函數
1.1 np_utils.to_categorical(y_label)
1.2 Sequential()
1.2.1 add()
裡面可放
1.2.1.1 Dense(units,input_dim,kernel_initializer='normal',activation, name,kernel_regularizer) 
1.2.1.2 Dropout(0.5)
1.2.1.3 Activation('relu')
1.2.1.4 Flatten()
1.2.1.5 Conv2D(filters, kernel_size, activation, padding='same')
1.2.1.6 Conv2D(filters=32,kernel_size=(3,3),input_shape=(32, 32,3), activation='relu', padding='same')
1.2.1.7 MaxPooling2D(pool_size)
1.2.1.8 Embedding(output_dim,input_dim,input_length)
1.2.1.9 SimpleRNN(units)
1.2.1.10 LSTM()
1.2.1.11 BatchNormalization()
1.2.2 summary()
1.2.3 compile(loss,optimizer,metrics=\['accuracy'\])
1.2.4 fit(x,y,validation_split,epochs,batch_size,verbose=2,shuffle,callbacks=\[earlystop or modelcheck or reduce_lr\])
1.2.4.1 history\['acc' or 'val_acc' or 'loss' or 'val_loss'\]
1.2.5 evaluate(x,y)輸出loss,accuracy
1.2.6 predict_classes(x)
1.3 keras.losses.mean_squared_error(y_true,y_pred)
1.4 keras.losses.categorical_crossentropy(y_true,y_pred)
1.5 keras.losses.hinge(y_true,y_pred)
1.6 keras.losses.binary _crossentropy(y_true,y_pred)
1.7 optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
1.8 optimizers. Adagrad(lr=0.01, epsilon=None, decay=0.0)
1.9 optimizers.RMSprop(lr=0.001, epsilon=None, decay=0.0)
1.10 optimizers. Adam(lr=0.001, epsilon=None, decay=0.0)
1.11 BatchNormalization()(x)
1.12 EarlyStopping(monitor="val_loss", patience=5, verbose=1)
1.13 ModelCheckpoint(filepath="./tmp.h5", monitor="val_loss", save_best_only=True)
1.14 keras.models.load_model(檔名)
1.15 ReduceLROnPlateau(factor=0.5, min_lr=1e-12, monitor='val_loss', patience=5, verbose=1)
1.16 Model(inputs=\[main_input, news_input\], outputs=\[main_output, news_output\])
1.16.1 compile(loss,optimizer,metrics=\['accuracy'\])
1.16.2 summary()
## 文字相關
2.1 Tokenizer(num_words)
2.1.1 fit_on_texts(train_text)
2.1.2 document_count(屬性)
2.1.3 word_index(屬性)
2.1.4 texts_to_sequences(train_text)
2.1.5 texts_to_sequences(test_text)
2.2 sequence.pad_sequences(x_train_seq,maxlen)
