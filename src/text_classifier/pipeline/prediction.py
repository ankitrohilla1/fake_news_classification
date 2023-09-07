from text_classifier.config.configuration import ConfigurationManager
from text_classifier.conponents.data_transformation import DataTransformation
import numpy as np
import os
from tensorflow.keras.models import load_model




class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    def predict(self,text):
        test_text = text['title'] + text['text']
        test_token = DataTransformation.convert_examples_to_features(test_text)
        model_path = os.path.join(self.config.model_path, 'bert-trained')
        # Load the pre-trained model
        model = load_model(model_path)
        test_text_pred = np.where(model.predict({ 'input_1' : test_token['input_ids'] , 'input_2' : test_token['attention_mask']}) >=0.5,1,0)

        if(test_text_pred[0]==0):
            output = "News is Fake"
        else:
            output = "News is Real"
        return output