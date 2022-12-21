import json
from flask import Flask
from CNN.m_predict import *
from decision_tree import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    token_label1, sam_label1 = load_data(url_full_train_data, "Sheet1", 300)
    label1_pred = predict('buồn', token_label1, sam_label1, load_aspect_model('CNN_train_3c_relu.json',
                                                                              'dts-phuclong_raw_train_2c-001-0.0144-1.0000.h5'),
                          labels)
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    if label1_pred == 'nev':
        mood = 1
    else:
        mood = 0
    y_pred_entropy = clf_entropy.predict([[1,0.8,0.8,1]])
    return json.dumps({'name': y_pred_entropy[0]})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
