from abc import ABC, abstractmethod
import pandas as pd
class BaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self, data):
        """
        Train the model on the given data.
        Must be implemented by concrete Model classes.
        """
        raise NotImplementedError("train method must be implemented") 
    
    @abstractmethod
    def prepare_features(self, data):
        """
        Prepare the features for the model.
        Must be implemented by concrete Model classes.
        """
        raise NotImplementedError("prepare_features method must be implemented") 
        

    def predict(self, data: pd.DataFrame):    
        """Predict the next step"""    
        # Prepare features    
        features = self.prepare_features(data)    
        features = features.dropna()    
        predictions = self.model.predict(features)    
        predictions = pd.DataFrame(index=features.index, data=predictions)    
        return predictions.to_dict()[0]
