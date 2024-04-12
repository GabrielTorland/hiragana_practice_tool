import pickle
from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
from utils import draw_strokes_to_image

class TMClassifier:
    def __init__(self, model_path, class_translation_table):
        self.class_translation_table = class_translation_table

        with open(model_path, 'rb') as file:
            self.model_state = pickle.load(file)

        self.tm = MultiClassConvolutionalTsetlinMachine2D(2000, 5000, 5.0, (10, 10))
        self.tm.__setstate__(self.model_state)

    def predict(self, strokes):
        image = draw_strokes_to_image(strokes, (28, 28), (800, 800))
        image = image.reshape(1, 28, 28)
        prediction = self.tm.predict(image)
        return self.class_translation_table[prediction[0]]