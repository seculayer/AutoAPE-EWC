from ewc.common.Constants import Constants

class ModelLoader(object):

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def model_loader(self):
        self.model.load_weights(f"{Constants.MODEL_RESOURCE_PATH}/{self.name}.h5")


