from flask import Flask, Response
import tensorflow as tf
from hand_recognition import check_hand_signs

app = Flask(__name__)
model = tf.keras.models.load_model('ipython/action.h5')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/video_feed')
def video_feed():
    return Response(check_hand_signs(), mimetype='multipart/x-mixed-replace; boundary=frame')

