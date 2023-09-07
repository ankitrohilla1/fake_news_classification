from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, load_metric
import torch
import pandas as pd
from tqdm import tqdm
from text_classifier.entity import ModelEvaluationConfig
from matplotlib import pyplot as plt
import os
import pickle
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras.models import load_model




class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def plot_model_accuracy(self, history):
        # summarize history for accuracy
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.config.graph, 'model_accuracy'))

    def plot_model_loss(self, history):
        # summarize history for modle loss
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.config.graph, 'model_loss'))


    def plot_confusion_matrix(self, y_test, y_pred):
        conf_matrix = confusion_matrix(y_test,y_pred)
        fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig(os.path.join(self.config.graph, 'confusion_matrix'))


    def check_performance(self):
        history_path = os.path.join(self.config.data_path, 'history.pkl')
        with open(os.path.join(history_path, 'rb')) as file:
            history = pickle.load(file)
        self.plot_model_accuracy(history)
        self.plot_model_loss(history)

        model_path = os.path.join(self.config.model_path, 'bert-trained')
        model = load_model(model_path)
        with open(os.path.join(self.config.data_path,'test_encodings.pkl'), 'rb') as file:
            loaded_data = pickle.load(file)
        X_test = loaded_data['X_test']
        y_test = loaded_data['y_test']
        y_pred = np.where(model.predict({ 'input_1' : X_test['input_ids'] , 'input_2' : X_test['attention_mask']}) >=0.5,1,0)
        self.plot_confusion_matrix(y_test, y_pred)

        

        


        

