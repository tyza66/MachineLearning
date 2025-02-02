import joblib
class MyModel:
    def __init__(self, model_name):
        self.new_new_attribute = model_name

if __name__ == '__main__':
    model = joblib.load('model.pkl')
    print(model.model_name)