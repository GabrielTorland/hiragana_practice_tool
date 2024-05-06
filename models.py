import pickle
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
from utils import draw_strokes_to_image
import torch
from torchvision import transforms

class Model:
    def load_model(self):
        raise NotImplementedError("load_model method must be implemented in child class")

    def predict(self):
        raise NotImplementedError("predict method must be implemented in child class")

class TMClassifier(Model):
    def __init__(self, model_path, class_translation_table):
        self.model_path = model_path
        self.class_translation_table = class_translation_table

    def load_model(self):
        self.model_state = pickle.load(open(self.model_path, "rb"))
        self.tm = MultiClassConvolutionalTsetlinMachine2D(10000, 5000, 5.0, (10, 10))
        self.tm.__setstate__(self.model_state)

    def predict(self, strokes):
        image = draw_strokes_to_image(strokes, (28, 28), (800, 800))
        image = image.reshape(1, 28, 28)
        prediction = self.tm.predict(image)
        return self.class_translation_table[prediction[0]]

class MobileNetClassifier(Model):
    def __init__(self, model_path, class_translation_table):
        self.model_path = model_path
        self.class_translation = class_translation_table
    
    def load_model(self):
        # use pytorch to load a pth model 
        self.model = torch.load(self.model_path, map_location=torch.device('cpu')) 

    def predict(self, strokes):
        image = draw_strokes_to_image(strokes, (28, 28), (800, 800))
        image = image.reshape(1, 28, 28)
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.permute(0, 2, 1, 3)
        prediction = self.model.forward(image)
        return self.class_translation[prediction[0]]
