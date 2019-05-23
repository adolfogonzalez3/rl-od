from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
import random
import os
import time
import math
import numpy as np
import numpy.random as npr
import gym
from gym import spaces
from keras.models import Model, Sequential
from keras import optimizers, regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout
import tensorflow as tf
import keras.backend as K
from randPic import *
import math
from keras.optimizers import SGD

from rl_od import utils


def neuralNet():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    model = Sequential()
    model.add(InputLayer((80, 80, 3)))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(2, activation='softmax'))

    return model


def randomTrain(self, epochs):
    # Pre-train the discriminator model
    for epoch in epochs:
        print("Train Epoch: ", epoch)

        '''
        train = []
        while(len(train) != len(self.test_images) // 5):
            x = np.random.randint(300)
            y = np.random.randint(300)
            img = random.choice(self.train_images)
            cropped = cropImage(img, x, y)
            if not(type(cropped) == int):
                train.append(cropped.copy())
        '''
        trainNp = np.array(self.fake.copy())
        idx = np.random.randint(len(self.test_images), size=len(self.fake))
        #idx = np.random.randint(len(self.test_images), size=len(trainNp))
        testNp = self.test_images[idx]
        #testNp = self.test_images.copy()
        x_train = np.row_stack((testNp, trainNp))
        y_labels = [1]*len(testNp) + [0]*len(trainNp)
        self.discriminator.fit(x=x_train, y=y_labels,
                               epochs=1, verbose=1, shuffle=True)
        evals = self.discriminator.evaluate(self.test_images_evaluate, y=[
                                            1]*len(self.test_images_evaluate))
        print(evals)
        evals = self.discriminator.evaluate(trainNp, y=[0]*len(trainNp))
        print(evals)
        trainEpoch -= 1
        if trainEpoch == 0:
            break


class rl_od(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.discriminator = neuralNet()
        self.discriminator.compile(
            optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

        cwd = os.path.dirname(__file__)
        self.data_path = "/home/adolfogonzaleziii/Desktop/rl_od/FaceData/"
        self.save_path = "/home/adolfogonzaleziii/Desktop/rl_od/rewards/"
        self.discPicNames = [f for f in listdir(
            self.data_path + "Disc/") if isfile(join(self.data_path + "Disc/", f))]
        self.agentPicNames = [f for f in listdir(
            self.data_path + "RL/") if isfile(join(self.data_path + "RL/", f))]

        self.disc_image = -1
        self.rl_image = -1
        self.rl_image_unflat = -1
        self.reward_average = []
        self.ind = 0
        self.chooser = 0
        self.x_val = -1
        self.y_val = -1
        self.observe = -1
        self.fake = []
        self.flag = 0

        self.goal = 0

        self.x_val_0 = -1
        self.x_val_1 = -1
        self.y_val_0 = -1
        self.y_val_1 = -1

        self.train_images = []
        for name in self.agentPicNames:
            image, _ = getRLImage(self.data_path, name)
            self.train_images.append(image / 255)
        self.train_images = np.array(self.train_images)

        self.test_images = []
        for name in self.discPicNames:
            image = getDiscImage(self.data_path, name)
            self.test_images.append(image / 255)
        self.test_images = np.array(self.test_images)

        idx = np.random.randint(len(self.test_images),
                                size=len(self.test_images) // 4)
        self.test_images_evaluate = self.test_images[idx]

        self.test_images = np.array(
            [e for i, e in enumerate(self.test_images) if i not in idx])

        '''
        for _ in range(5):
            randomTrain(self, 2)
        '''

        self.counter = 0
        self.action_space = spaces.Discrete(300)
        self.observation_space = spaces.Box(0, 1, shape=(225, 300, 4),
                                            dtype=np.float)

    def step(self, action):
        if self.chooser == 0:
            self.chooser = 1
            terminal = False
            reward = 0.01
            if action >= 225:
                self.x_val_0 = 224
            else:
                self.x_val_0 = action

        elif self.chooser == 1:
            self.chooser = 2
            terminal = False
            reward = 0.01
            if action >= 225:
                self.x_val_1 = 225
            else:
                self.x_val_1 = action

        elif self.chooser == 2:
            self.chooser = 3
            terminal = False
            reward = 0.01
            self.y_val_0 = action
        else:
            self.chooser = 0
            terminal = True
            self.y_val_1 = action
            pic, cropped = cropper(
                self.rl_image_unflat, self.x_val_0, self.x_val_1, self.y_val_0, self.y_val_1)
            reward = -0.01
            if not(type(cropped) == int) and not(type(cropped) == float):
                fake_image = np.expand_dims(cropped.copy(), axis=0)

                if self.goal == 0:
                    reward = 1 - self.discriminator.predict(fake_image)[0][0]
                    self.goal = 1
                    self.fake.append(np.squeeze(fake_image))
                else:
                    reward = self.discriminator.predict(fake_image)[0][0]
                    self.fake.append(np.squeeze(fake_image))
                    self.goal = 0

                self.ind += 1

                if len(self.fake) == 100:
                    drawRect(pic.copy(), self.x_val, self.y_val,
                             self.ind, False, self.goal)
                    randomTrain(self, 1)
                    self.fake = []

                if reward >= 0.5:
                    drawRect(pic.copy(), self.x_val, self.y_val,
                             self.ind, True, self.goal)
                if self.goal == 0:
                    self.reward_average.append(reward)
                if len(self.reward_average) == 10:
                    message = '\n RL Avg: (r/10): {:+.3f} Iter: {:+.3f}\n ---------------'
                    avg = np.average(self.reward_average)
                    message = message.format(avg, self.ind)
                    self.reward_average = []
                    print(message)
                self.observe = paster(pic.copy())

        return self.observe, reward, terminal, {}

    def reset(self):
        self.disc_image = random.choice(self.test_images)
        self.observe = random.choice(self.train_images)
        self.rl_image_unflat = self.observe.copy()
        if self.goal == 0:
            self.observe = np.append(
                self.observe, np.zeros((225, 300, 1)), axis=-1)
        else:
            self.observe = np.append(
                self.observe, np.ones((225, 300, 1)), axis=-1)
        self.x_val = -1
        self.y_val = -1
        self.chooser = 0
        if self.ind >= 400:
            self.ind = 0
        return self.observe

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass
