from text_classifier.entity import ModelTrainerConfig
import torch
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,Input
from transformers import TFBertModel 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pickle


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def create_model(self, bert_model):
        input_ids=Input(shape=(self.config.maxlen,),dtype=tf.int32)
        input_mask=Input(shape=(self.config.maxlen,),dtype=tf.int32)
        bert_layer=bert_model([input_ids,input_mask])[1]
        x=Dropout(0.5)(bert_layer)
        x=Dense(64,activation="tanh")(x)
        x=Dropout(0.2)(x)
        x=Dense(1,activation="sigmoid")(x)
        model = Model(inputs=[input_ids, input_mask], outputs=x)
        return model
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model = TFBertModel.from_pretrained(self.config.model_ckpt)
        with tf.device(device):
            model = self.create_model(bert_model)
        print(model.summary())
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-08,clipnorm=1.0)
        model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = self.config.metrics)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='max', verbose=1, 
                                                    patience=50,baseline=0.4,min_delta=0.0001,
                                                    restore_best_weights=False)
        with open(os.path.join(self.config.data_path,'train_encodings.pkl'), 'rb') as file:
            loaded_data = pickle.load(file)
        X_train = loaded_data['X_train']
        y_train = loaded_data['y_train']
        
        history = model.fit(x = {'input_1':X_train['input_ids'],'input_2':X_train['attention_mask']}, 
                            y = y_train, epochs=self.config.n_epochs, validation_split = 0.2, 
                            batch_size = 30, callbacks=[callback])
       
        ## Save model
        # model.save_pretrained(os.path.join(self.config.root_dir,"bert-news-classify-model.pkl"))

        # Save history
        # with open(os.path.join(self.config.root_dir,'history.pkl'), 'wb') as file:
        #     pickle.dump(history, file)

