class Model:
    def __init__(self):
        self.__layer = []

    def add(self, layer):
        self.__layer.append(layer)

    def predict(self, x_list):
        for x in x_list:
            for i in range(len(self.__layer)):
                x = y = self.__layer[i].forward_pass(x)
        return y

    def update(self, E):
        for i in range(len(self.__layer)):
            E = self.__layer[i].back_propagate(E)

    def train(self, x_list, y_list):
        for x, y in x_list, y_list:
            E = self.predict(x) - y
            self.update(E)
