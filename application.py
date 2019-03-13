from flask import Flask, request,  render_template
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pyrebase
import time

app = Flask(__name__)

#variables for the model like features and steps to take
N_CLASSES = 4
N_HIDDEN_UNITS = 64
N_TIME_STEPS = 15
N_FEATURES = 6

training= True
is_predict= False

step = 1
segments = []
labels1 = []

column_names = ["g_x", "g_y", "g_z", "a_x", "a_y", "a_z","target"]
df = pd.read_csv("zero_data_v1.csv", names = column_names)
target = df.target

for i in range(0, len(df) - N_TIME_STEPS, step):
    a_x = df['a_x'].values[i: i + N_TIME_STEPS]
    a_y = df['a_y'].values[i: i + N_TIME_STEPS]
    a_z = df['a_z'].values[i: i + N_TIME_STEPS]
    g_x = df['g_x'].values[i: i + N_TIME_STEPS]
    g_y = df['g_y'].values[i: i + N_TIME_STEPS]
    g_z = df['g_z'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['target'][i: i + N_TIME_STEPS])[0][0]
    segments.append([g_x, g_y, g_z,a_x, a_y, a_z,])
    labels1.append(label)

segments= np.swapaxes(segments,1,2)
reshaped_segments=np.asarray(segments, dtype=np.float32)
labels= np.asarray(pd.get_dummies(labels1), dtype = np.float32)

# initialize firebase
config = {
    "apiKey": "AIzaSyDIAFBwmKmGt-B2RPBcl-ZjHuRGo-R1JZ0",
    "authDomain": "gix-510.firebaseapp.com",
    "databaseURL": "https://gix-510.firebaseio.com",
    "projectId": "gix-510",
    "storageBucket": "gix-510.appspot.com",
    "messagingSenderId": "721539835802"
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth();
db = firebase.database()

def data_receiving():
    dataset = db.get().val()
    return dataset

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

#run this section just one time
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])
pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")
L2_LOSS = 0.0015
l2 = L2_LOSS * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

LEARNING_RATE = 0.005
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
#end of setup for LSTM model

#some one time variable setup
motions = ['not cooking', 'cutting', 'whisking', 'saute']

model_train=False
dataset = data_receiving()
dataset = dataset['dataset']
prev_data = np.asarray(dataset, dtype=np.float32)

#running the while loop starts here
prev_step =- 1
step =- 1

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/')
def infinite_loop():
    global prev_step, prev_data, reshaped_segments, labels1
    while 1:
        # check the marker we get from frontend
        # 0 is do nothing
        # 1 is training
        # 2 is predict
        dataset = data_receiving()
        step = dataset['step']
        marker = dataset['marker']

        if marker == 0:
            training = False
            is_predict = False

        elif marker == 1:
            training = True
            is_predict = False

        elif marker == 2:
            training = False
            is_predict = True
        # check to see if it is training AND collecting data
        # if it is collecting data, add it to the bottom of our current data
        if training == True and step <= 405 and step != -1:
            dataset = data_receiving()
            dataset = dataset['dataset']
            dataset = np.asarray(dataset, dtype=np.float32)

            if step != prev_step and dataset[0][0][0] != prev_data[0][0][0]:
                print(step)
                feature_label = dataset[:, :, 6]
                feature_label = feature_label[0][0]
                feature_data = np.delete(dataset, 6, 2)
                reshaped_segments = np.concatenate((reshaped_segments, feature_data), axis=0)
                labels1.append(feature_label)
                prev_step = step
        # once you are finished training the model
        if training == True and step >= 405 and step != -1:
            #         print(step)
            #         print(labels1.shape)
            print(reshaped_segments.shape)
            print(len(labels1))
            labels = np.asarray(pd.get_dummies(labels1), dtype=np.float32)

            X_train, X_test, y_train, y_test = train_test_split(
                reshaped_segments, labels, test_size=0.2, random_state=20)
            #         print(X_train.shape)
            #         print(X_test.shape)
            N_EPOCHS = 150
            BATCH_SIZE = 128

            tf.reset_default_graph()
            X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
            Y = tf.placeholder(tf.float32, [None, N_CLASSES])
            pred_Y = create_LSTM_model(X)
            pred_softmax = tf.nn.softmax(pred_Y, name="y_")

            l2 = L2_LOSS * \
                 sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
            correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

            saver = tf.train.Saver()

            history = dict(train_loss=[],
                           train_acc=[],
                           test_loss=[],
                           test_acc=[])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            train_count = len(X_train)

            for i in range(1, N_EPOCHS + 1):
                for start, end in zip(range(0, train_count, BATCH_SIZE),
                                      range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                    sess.run(optimizer, feed_dict={X: X_train[start:end],
                                                   Y: y_train[start:end]})

                _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
                    X: X_train, Y: y_train})

                _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
                    X: X_test, Y: y_test})

                history['train_loss'].append(loss_train)
                history['train_acc'].append(acc_train)
                history['test_loss'].append(loss_test)
                history['test_acc'].append(acc_test)

                if i != 1 and i % 10 != 0:
                    continue

                print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')
                # saves the final version of the model
                # which creates 4 files!!!
                if (i == N_EPOCHS):
                    saver.save(sess, 'LSTM_15step', global_step=i)
            predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss],
                                                          feed_dict={X: X_test, Y: y_test})

            print()
            print(f'final results: accuracy: {acc_final} loss: {loss_final}')
            sess.close()
            db.update({'train_done': 1})
        # end of model training.

        # predictions section here
        if is_predict == True:
            print("starting predictions")
            with tf.Session() as sess2:
                new_saver = tf.train.import_meta_graph('LSTM_15step-150.meta')
                new_saver.restore(sess2, tf.train.latest_checkpoint('./'))
                currtime = time.time()
                move = -1
                prev_preds = [0, 0, 0]
                dataset = data_receiving()
                dataset = dataset['dataset']
                prev_data = np.asarray(dataset, dtype=np.float32)
                while True:
                    newtime = time.time()
                    dataset = data_receiving()
                    keep_predicting = dataset['marker']

                    # check to see if you keep predicting or not
                    if (keep_predicting != 2):
                        print("predictions ended")
                        reshaped_segments = reshaped_segments[:685, :, :]
                        labels1 = labels1[:685]
                        time.sleep(2)
                        sess2.close()
                        break
                    dataset = dataset['dataset']
                    dataset = np.asarray(dataset, dtype=np.float32)

                    if (dataset[0][0][0] == prev_data[0][0][0]):
                        continue

                    predictions2 = sess2.run(pred_softmax, feed_dict={X: dataset})
                    prev_preds.append(np.argmax(predictions2))
                    if (np.argmax(predictions2) != move):
                        if prev_preds.count(np.argmax(predictions2)) > 3:  # CHANGE NUMBER FOR MORE OR LESS FLEXIBILITY
                            print("Current motion is:", motions[np.argmax(predictions2)])
                            move = np.argmax(predictions2)
                            db.update({'current_prediction':str(move)})
                            prev_preds.clear()
                            prev_preds.append(np.argmax(predictions2))
                            prev_preds.append(np.argmax(predictions2))
                            prev_preds.append(np.argmax(predictions2))
                            prev_preds.append(np.argmax(predictions2))
                    prev_preds.pop(0)
                    prev_data = dataset
                    time.sleep(0.1)
    return "It's running"

if __name__ == "__main__":
    print("Start running!")
    print("I'm so excited!!!!")
    app.run(host="127.0.0.1", port=8080, debug=True)
    # app.run(host='https://test-n-one-smart-kitchen.azurewebsites.net', port=8000, debug=True)