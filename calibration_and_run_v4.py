import smbus
import math
import time
import pandas as pd
import numpy as np
import os
import pyrebase

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

power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

def data_receiving():
  dataset = db.get().val()
  return dataset

def read_byte(reg):
    return bus.read_byte_data(address, reg)


def read_word(reg):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg + 1)
    value = (h << 8) + l
    return value


def read_word_2c(reg):
    val = read_word(reg)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val


def dist(a, b):
    return math.sqrt((a * a) + (b * b))


def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)


def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


step=0
window_size=15

#for training data
def collecting(target,step):
    # target = 0
    output = []
    dataset = []
    start = time.time()

    for i in range(150):
        print("Gyroscope")
        print("--------")
        gyroskop_xout = read_word_2c(0x43)
        gyroskop_yout = read_word_2c(0x45)
        gyroskop_zout = read_word_2c(0x47)

        gy_x = np.divide(gyroskop_xout, 131)
        gy_y = np.divide(gyroskop_yout, 131)
        gy_z = np.divide(gyroskop_zout, 131)

        print("gyroscope_xout: ", ("%5d" % gyroskop_xout), " scale: ", gy_x)
        print("gyroscope_yout: ", ("%5d" % gyroskop_yout), " scale: ", gy_y)
        print("gyroscope_zout: ", ("%5d" % gyroskop_zout), " scale: ", gy_z)

        print("Acceleration")
        print("---------------------")

        beschleunigung_xout = read_word_2c(0x3b)
        beschleunigung_yout = read_word_2c(0x3d)
        beschleunigung_zout = read_word_2c(0x3f)

        beschleunigung_xout_skaliert = np.divide(beschleunigung_xout, 16384.0)
        beschleunigung_yout_skaliert = np.divide(beschleunigung_yout, 16384.0)
        beschleunigung_zout_skaliert = np.divide(beschleunigung_zout, 16384.0)

        print("accel_xout: ", ("%6d" % beschleunigung_xout), " scale: ", beschleunigung_xout_skaliert)
        print("accel_yout: ", ("%6d" % beschleunigung_yout), " scale: ", beschleunigung_yout_skaliert)
        print("accel_zout: ", ("%6d" % beschleunigung_zout), " scale: ", beschleunigung_zout_skaliert)

        print("X Rotation: ",
              get_x_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert))
        print("Y Rotation: ",
              get_y_rotation(beschleunigung_xout_skaliert, beschleunigung_yout_skaliert, beschleunigung_zout_skaliert))

        data = [gy_x, gy_y, gy_z, beschleunigung_xout_skaliert, beschleunigung_yout_skaliert,
                beschleunigung_zout_skaliert, target]
        dataset.append(data)
        # time.sleep(0.2)
        start = time.time()
        if len(dataset)<=window_size:
            time.sleep(0.2)
        elif len(dataset) > window_size:
            dataset.pop(0)
            step = step + 1
            print(step)
            # db.update({"dataset": [dataset]})
            db.update({'dataset':[dataset],'step':step})
        # print("total time: ", time.time() - start)
    return dataset, step

#for predicting data
def collecting_predict():
    # target = 0
    # output = []
    dataset = []

    ######  collect data once in a while  ####

    ##########################################
    flag = True
    # i=1
    while (flag):
    # while(i==1):
        # start = time.time()
        gyroskop_xout = read_word_2c(0x43)
        gyroskop_yout = read_word_2c(0x45)
        gyroskop_zout = read_word_2c(0x47)
        beschleunigung_xout = read_word_2c(0x3b)
        beschleunigung_yout = read_word_2c(0x3d)
        beschleunigung_zout = read_word_2c(0x3f)
        # start=time.time()
        # print("time for each update: ", time.time() - start)
        beschleunigung_xout_skaliert = np.divide(beschleunigung_xout, 16384.0)
        beschleunigung_yout_skaliert = np.divide(beschleunigung_yout, 16384.0)
        beschleunigung_zout_skaliert = np.divide(beschleunigung_zout, 16384.0)
        gy_x = np.divide(gyroskop_xout, 131)
        gy_y = np.divide(gyroskop_yout, 131)
        gy_z = np.divide(gyroskop_zout, 131)
        # print("time for each update: ", time.time() - start)

        data = [gy_x, gy_y, gy_z, beschleunigung_xout_skaliert, beschleunigung_yout_skaliert,
                beschleunigung_zout_skaliert]

        dataset.append(data)
        start = time.time()
        if len(dataset) > window_size:
            dataset.pop(0)
            db.update({"dataset": [dataset]})
        print("time for each update: ", time.time() - start)
        if len(dataset) <= window_size:
            time.sleep(0.2)
        # r = requests.post("http://205.175.106.102:5000/", json={"data":dataset})
        # r = requests.post("0.0.0.0", json={"data": dataset})
    # time.sleep(0.1)
    return dataset


bus = smbus.SMBus(1)  # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68  # via i2cdetect

# Aktivieren, um das Modul ansprechen zu koennen
bus.write_byte_data(address, power_mgmt_1, 0)
process = time.time()

training = True
predict= True
while(1):
    start=time.time()
    my_data= data_receiving()
    # mark=my_data['marker']
    # if(mark==0):
    #     training=False
    #     predict=False
    # if(mark==1):
    #     training=True
    #     predict=False
    # if (mark==2):
    #     training=False
    #     predict=True
    if(training==True):
        ## MOTION1 - cutting
        print("You have three seconds to prepare for next motion: cutting")
        time.sleep(1)
        print('2')
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("start cutting!")
        data, step = collecting(1,step)
        print("stop cutting")
        time.sleep(1)
        ## MOTION2 - whisking
        print("You have three seconds to prepare for next motion: whisking")
        time.sleep(1)
        print('2')
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("start whisking!")
        data, step = collecting(2,step)
        print("stop whisking")
        time.sleep(1)
        print("You have three seconds to prepare for next motion: stir frying/saute")
        time.sleep(1)
        print('2')
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("start stir frying!")
        data, step = collecting(3,step)
        print("stop stir frying")
        time.sleep(1)
        training=False

    elif(predict==True):
        training_done= data_receiving()
        training_done= training_done['train_done']
        if(training_done==1):
            db.update({'train_done':0})
            dataset = collecting_predict()
        # predict=False
    print("time per while loop is:", time.time()-start)
