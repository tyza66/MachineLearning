import joblib
class MyModel:
    def __init__(self, model_name, new_attribute):
        self.new_attribute = new_attribute
        self.model_name = model_name
        self.new_new_attribute = 'new_new_attribute'

if __name__ == '__main__':
    model = joblib.load('model.pkl')
    print(model.model_name)