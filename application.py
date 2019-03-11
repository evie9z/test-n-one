from flask import Flask, request
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pyrebase
import time

app = Flask(__name__)

def firebase_initialization():
    # code below sets up pyrebase and receiving function
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
    database = firebase.database()
    return database

def load_zero_data():
    # below reads the csv and sets up the data in the right format
    column_names = ["g_x", "g_y", "g_z", "a_x", "a_y", "a_z", "target"]
    df = pd.read_csv("zero_data_v1.csv", names=column_names)
    target = df.target
    step = 1
    segments = []
    labels1 = []

    for i in range(0, len(df) - n_time_steps, step):
        a_x = df['a_x'].values[i: i + n_time_steps]
        a_y = df['a_y'].values[i: i + n_time_steps]
        a_z = df['a_z'].values[i: i + n_time_steps]
        g_x = df['g_x'].values[i: i + n_time_steps]
        g_y = df['g_y'].values[i: i + n_time_steps]
        g_z = df['g_z'].values[i: i + n_time_steps]
        label = stats.mode(df['target'][i: i + n_time_steps])[0][0]
        segments.append([g_x, g_y, g_z, a_x, a_y, a_z, ])
        labels1.append(label)

    segments = np.swapaxes(segments, 1, 2)
    reshaped_segments = np.asarray(segments, dtype=np.float32)
    labels = np.asarray(pd.get_dummies(labels1), dtype=np.float32)
    return reshaped_segments, labels1

def create_lstm_model(inputs):
    # variables for the model like features and steps to take
    n_classes = 4
    n_hidden_units = 64
    n_time_steps = 15
    n_features = 6

    W = {
        'hidden': tf.Variable(tf.random_normal([n_features, n_hidden_units])),
        'output': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden_units], mean=1.0)),
        'output': tf.Variable(tf.random_normal([n_classes]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, n_features])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, n_time_steps, 0)

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

@app.route('/')
def data_receiving():
    dataset = db.get().val()
    return dataset

@app.route('/train')
def infinite_loop():
    # check the marker we get from frontend
    # 0 is do nothing
    # 1 is training
    # 2 is predict
    training = False
    is_predict = False
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
    print(training, step)
    # check to see if it is training AND collecting data
    # if it is collecting data, add it to the bottom of our current data
    if training == True and step <= 405 and step != -1:
        dataset = data_receiving()
        dataset = dataset['dataset']
        dataset = np.asarray(dataset, dtype=np.float32)

        if (dataset[0][0][0] != prev_data[0][0][0]) and (step != prev_step):
            print(step)
            feature_label = dataset[:, :, 6]
            feature_label = feature_label[0][0]
            feature_data = np.delete(dataset, 6, 2)
            reshaped_segments = np.concatenate((reshaped_segments, feature_data), axis=0)
            labels1.append(feature_label)
            prev_step = step

    # once you are finished training the model
    if training == True and step >= 405 and step != -1:

        print(step)
        labels = np.asarray(pd.get_dummies(labels1), dtype=np.float32)
        print(labels.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, labels, test_size=0.2, random_state=20)
        n_epochs = 150
        batch_size = 128

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, n_time_steps, n_features], name="input")
        Y = tf.placeholder(tf.float32, [None, n_classes])
        pred_Y = create_lstm_model(X)
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

        for i in range(1, n_epochs + 1):
            for start, end in zip(range(0, train_count, batch_size),
                                  range(batch_size, train_count + 1, batch_size)):
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
            if i == n_epochs:
                saver.save(sess, 'LSTM_15step', global_step=i)
        predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

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
            #   print(sess2.run(pred_softmax, feed_dict={X: dataset}))
            prev_data = np.asarray(dataset, dtype=np.float32)
            while True:
                newtime = time.time()
                dataset = data_receiving()
                keep_predicting = dataset['marker']

                # check to see if you keep predicting or not
                if keep_predicting != 2:
                    print("predictions ended")
                    time.sleep(2)
                    sess2.close()
                    break
                dataset = dataset['dataset']
                dataset = np.asarray(dataset, dtype=np.float32)

                if dataset[0][0][0] == prev_data[0][0][0]:
                    continue

                predictions2 = sess2.run(pred_softmax, feed_dict={X: dataset})
                prev_preds.append(np.argmax(predictions2))
                if np.argmax(predictions2) != move:
                    if prev_preds.count(np.argmax(predictions2)) > 3:  # CHANGE NUMBER FOR MORE OR LESS FLEXIBILITY
                        print("Current motion is:", motions[np.argmax(predictions2)])
                        move = np.argmax(predictions2)
                        prev_preds.clear()
                        prev_preds.append(np.argmax(predictions2))
                        prev_preds.append(np.argmax(predictions2))
                        prev_preds.append(np.argmax(predictions2))
                        prev_preds.append(np.argmax(predictions2))

                prev_preds.pop(0)
                prev_data = dataset
                time.sleep(0.1)

@app.route('/test')
def test():
    return "It's running"


if __name__ == "__main__":

    db = firebase_initialization()
    reshaped_segments, labels1 = load_zero_data()

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, n_time_steps, n_features], name="input")
    Y = tf.placeholder(tf.float32, [None, n_classes])
    pred_Y = create_lstm_model(X)
    pred_softmax = tf.nn.softmax(pred_Y, name="y_")
    L2_LOSS = 0.0015
    l2 = L2_LOSS * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

    LEARNING_RATE = 0.005
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # some one time variable setup
    motions = ['not cooking', 'cutting', 'whisking', 'saute']

    model_train = False
    dataset = data_receiving()
    dataset = dataset['dataset']
    prev_data = np.asarray(dataset, dtype=np.float32)

    # running the while loop starts here
    prev_step = - 1
    step = - 1

    app.run()