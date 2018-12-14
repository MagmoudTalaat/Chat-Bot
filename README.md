Chat-Bot

This is a chat bot built using data from ChatterBot found in the link: https://github.com/gunthercox/ChatterBot licensed under the BSD 3-clause license. This dataset contains possible messages and the corresponding replies. Each group of messages  related to a specific conversation type (genre) are stored in one file named with that genre. This bot predicts which genre the user message belongs to then predicts which response best suits the message from that genre. This bot replies are independent of the previous conversation.

Description of python classes:

genre-classification:
This class trains the model that classifies the messages sent by the user into one of the genres which are stored in the documents stored in the data folder. The classification is done using Doc2Vec model and the model is stored in the model folder.

doc-classification:
This class trains the model that determines to which sentence the user message is similar to using Doc2Vec model and stores the model in the model folder.

chatbot:
This class calls both classifiers to determine the best response to the user message.

Server:
This is the flask socketio server which is responsible for recieving the user messages and sending the replies.

Description of folders:

data:
This folder contains the dataset files.

model:
This file contains the trained models for genre classification model and document classification model.

template:
This folder contains index.html file which is the web interface for the bot.
