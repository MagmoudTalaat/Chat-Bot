from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit
import chatbot
#initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
genre_model=chatbot.load_genre_model()
user_messages="" #trying to get all user messages and classifying the entire messages to keep conversation in context and bring smooth changes between topics
#on connection event
@socketio.on('connect')
def test_connect():
	print("client connected")
	emit('connected', {'data': 'Connected'})
#on message event
@socketio.on('message')
def message_rec(msg):
	#global user_messages
	print("message received: "+msg)
	#user_messages+=msg+"\n" #a try to bring a conversation into a certain context with smoother changes between topics
	reply=chatbot.get_response(msg,genre_model)
	#user_messages+=reply+"\n" #a try to bring a conversation into a certain context with smoother changes between topics
	emit('reply', reply)

#load index
@app.route('/')
def load_index():
	return render_template('index.html')
def messageReceived(methods=['Get','POST']):
	print('message received')
if __name__ == '__main__':
    socketio.run(app,port=3000)