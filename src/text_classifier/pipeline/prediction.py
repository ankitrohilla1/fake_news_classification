from text_classifier.config.configuration import ConfigurationManager
from text_classifier.conponents.data_transformation import DataTransformation
import numpy as np


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        test_text = text['text'] + text['title']
        tokenizer = DataTransformation.convert_examples_to_features(test_text)
        model = self.config.model_path
        test_token = self.config.test_encodings
        test_text_pred = np.where(model.predict({ 'input_1' : test_token['input_ids'] , 'input_2' : test_token['attention_mask']}) >=0.5,1,0)

        if(test_text_pred[0]==0):
            output = "News is Fake"
        else:
            output = "News is Real"
        return output