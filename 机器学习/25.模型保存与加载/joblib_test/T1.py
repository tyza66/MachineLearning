import joblib
class MyModel:
    def __init__(self, model_name):
        self.model_name = model_name

if __name__ == '__main__':
    model = MyModel('model1')
    joblib.dump(model, 'model.pkl')