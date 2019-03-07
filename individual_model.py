import pandas as pd
import numpy as np
# import pickle
# import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
# from pylab import rcParams
# from sklearn import metrics
from sklearn.model_selection import train_test_split
import pyrebase
import time
column_names = ["g_x", "g_y", "g_z", "a_x", "a_y", "a_z","target"]
df = pd.read_csv("codycooking1.csv", names = column_names)
target = df.target

#time steps is based on how many time points you want to put into one entry
N_TIME_STEPS = 15
#number of features, should be 6 since we have acceleration and gyroscope for xyz
N_FEATURES = 6
step = 1
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    a_x = df['a_x'].values[i: i + N_TIME_STEPS]
    a_y = df['a_y'].values[i: i + N_TIME_STEPS]
    a_z = df['a_z'].values[i: i + N_TIME_STEPS]
    g_x = df['g_x'].values[i: i + N_TIME_STEPS]
    g_y = df['g_y'].values[i: i + N_TIME_STEPS]
    g_z = df['g_z'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['target'][i: i + N_TIME_STEPS])[0][0]
    segments.append([g_x, g_y, g_z,a_x, a_y, a_z,])
    labels.append(label)
segments= np.swapaxes(segments,1,2)
reshaped_segments=np.asarray(segments, dtype=np.float32)
print(np.array(segments).shape)
print(segments[0])
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=20)
print(X_train.shape)
print(X_test.shape)

N_CLASSES = 4
N_HIDDEN_UNITS = 64


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

    # Stack 2 LSTM layers
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])
pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")
L2_LOSS = 0.0015
l2 = L2_LOSS * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = Y)) + l2
LEARNING_RATE = 0.005
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

##training
N_EPOCHS = 250
BATCH_SIZE=128
saver = tf.train.Saver()

history = dict(train_loss=[],
                     train_acc=[],
                     test_loss=[],
                     test_acc=[])
sess=tf.Session()
sess.run(tf.global_variables_initializer())

train_count = len(X_train)

for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
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
    #saves the final version of the model
    #which creates 4 files!!!
    if(i==N_EPOCHS):
      saver.save(sess, 'LSTM_15step', global_step=i)
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')
sess.close()

config = {

    "apiKey": "AIzaSyDIAFBwmKmGt-B2RPBcl-ZjHuRGo-R1JZ0",
    "authDomain": "gix-510.firebaseapp.com",
    "databaseURL": "https://gix-510.firebaseio.com",
    "projectId": "gix-510",
    "storageBucket": "gix-510.appspot.com",
    "messagingSenderId": "721539835802"
  }
firebase = pyrebase.initialize_app(config)

auth= firebase.auth();
# user= auth.sign_in_anonymous();
db = firebase.database()
def data_receiving():
  dataset = db.get().val()
  return dataset



motions = ['not cooking', 'cutting', 'whisking', 'saute', 'season']
with tf.Session() as sess2:
    new_saver = tf.train.import_meta_graph('LSTM_15step-250.meta')
    new_saver.restore(sess2, tf.train.latest_checkpoint('./'))
    currtime = time.time()
    marker = -1
    prev_preds = [0, 0, 0, 0, 0]
    dataset = data_receiving()
    dataset = dataset['dataset']
    #   print(sess2.run(pred_softmax, feed_dict={X: dataset}))
    prev_data = np.asarray(dataset, dtype=np.float32)
    while (True):
        newtime = time.time()
        dataset = data_receiving()
        dataset = dataset['dataset']
        dataset = np.asarray(dataset, dtype=np.float32)
        #     print(dataset.shape)
        if (dataset[0][0][0] == prev_data[0][0][0]):
            continue
        #     print(dataset)
        predictions2 = sess2.run(pred_softmax, feed_dict={X: dataset})
        # print(motions[np.argmax(predictions2)])

        prev_preds.append(np.argmax(predictions2))
        if (np.argmax(predictions2) != marker):
            #       print("Current motion is:",np.argmax(predictions2))
            if (prev_preds.count(np.argmax(predictions2)) > 5):  # CHANGE NUMBER FOR MORE OR LESS FLEXIBILITY
                print("Current motion is:", motions[np.argmax(predictions2)])
                marker = np.argmax(predictions2)
                prev_preds.clear()
                prev_preds.append(np.argmax(predictions2))
                prev_preds.append(np.argmax(predictions2))
                prev_preds.append(np.argmax(predictions2))
                prev_preds.append(np.argmax(predictions2))
                prev_preds.append(np.argmax(predictions2))
                prev_preds.append(np.argmax(predictions2))
        prev_preds.pop(0)
        prev_data = dataset
        time.sleep(0.1)