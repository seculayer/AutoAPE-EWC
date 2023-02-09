from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist

class DataLoader(object):

    @staticmethod
    def load_data(data_type, model_type):

        x_train = list() ; y_train = list() ; x_test = list(); y_test = list()

        if data_type.lower()=='written':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif data_type.lower()=='fashion':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            'data_type is not valid. please check The "data_type"'

        img_rows, img_cols = x_train.shape[1:]
        if model_type.lower()=='cnn':
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
        elif model_type.lower()=='fdd':
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        else:
            'model_type is not valid. please check The "model_type"'

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    data_category='Written'
    model = 'cnn'

    x_data, x_label, y_data, y_label = DataLoader().load_data(data_category, model)
