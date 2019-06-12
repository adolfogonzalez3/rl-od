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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout
import tensorflow as tf
import tensorflow.keras.backend as K
from randPic import *
import math
from tensorflow.keras.optimizers import SGD

from rl_od import utils
from rl_od.data import load_data


def neuralNet():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    model = Sequential()
    model.add(InputLayer((80, 80, 1)))

    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


class rl_od(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.discriminator = neuralNet()
        self.discriminator.compile(
            optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

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

        self.train_images, test_images = load_data()
        self.train_images = self.train_images.astype(np.float) / 255
        test_images = test_images.astype(np.float) / 255

        npr.shuffle(test_images)
        self.test_images_evaluate = test_images[:(len(test_images) // 4)]
        self.test_images = test_images[(len(test_images) // 4):]
        #self.train_images = self.test_images

        '''
        for _ in range(5):
            randomTrain(self, 2)
        '''
        self.choices = []

        self.counter = 0
        #self.action_space = spaces.Discrete(80)
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 1, shape=(80, 80, 1),
                                            dtype=np.float)
        # self.observation_space = spaces.Box(0, 1, shape=(225, 300, 4),
        #                                    dtype=np.float)

    def step(self, action):
        terminal = False
        reward = -0.1
        if action != 5 and self.timestep < 3:
            if action == 0:
                upper, left, bottom, right = (0, 0, 50, 50)
            elif action == 1:
                upper, left, bottom, right = (0, 50, 50, 80)
            elif action == 2:
                upper, left, bottom, right = (30, 0, 80, 30)
            elif action == 3:
                upper, left, bottom, right = (30, 30, 80, 80)
            elif action == 4:
                upper, left, bottom, right = (15, 15, 65, 65)
            sub_img = utils.crop_array(self.observe, upper, left, bottom,
                                       right)
            converted = (sub_img*255).astype(np.uint8)
            converted = np.squeeze(converted, axis=-1)
            converted_img = Image.fromarray(converted)
            resized_img = converted_img.resize((80, 80), Image.ANTIALIAS)
            array_img = np.array(resized_img, dtype=np.float) / 255
            self.observe = np.expand_dims(array_img, axis=-1)
        else:
            terminal = True
        self.fake.append(self.observe)
        fake_image = np.expand_dims(self.observe, axis=0)
        pred = float(self.discriminator.predict_on_batch(fake_image))
        reward = pred - self.last_pred
        self.last_pred = pred
        #reward = reward - 1.
        if len(self.fake) == 1000:
            print('saving...')
            array_img = np.squeeze(self.observe, axis=-1)
            path = "./pics1/%d-%d.jpg" % (self.ind, int(self.goal))
            Image.fromarray((array_img*255).astype(np.uint8)).save(path)
            self.randomTrain()
            self.fake = []
            print('done!')
        self.ind += 1
        self.reward_average.append(reward)
        if len(self.reward_average) == 10:
            message = '\n RL Avg: (r/10): {:+.3f} Iter: {:+.3f}\n ---------------'
            avg = np.average(self.reward_average)
            message = message.format(avg, self.ind)
            self.reward_average = []
            print(message)
        self.timestep += 1
        return self.observe, reward, terminal, {}

        if len(self.choices) < 3:
            self.choices.append(action)
        else:
            terminal = True
            self.choices.append(action)
            cropped_img = utils.crop_array(self.rl_image_unflat, *self.choices)
            if cropped_img.size == 0:
                cropped_img = self.rl_image_unflat
            converted = (cropped_img*255).astype(np.uint8)
            converted = np.squeeze(converted, axis=-1)
            #print(converted.dtype, converted.shape)
            cropped_img = Image.fromarray(converted)
            cropped_img = cropped_img.resize((80, 80), Image.ANTIALIAS)
            cropped_img = np.array(cropped_img, dtype=np.float) / 255
            fake_image = np.expand_dims(cropped_img, axis=-1)
            self.fake.append(fake_image)
            fake_image = np.expand_dims(fake_image, axis=0)
            reward = float(self.discriminator.predict_on_batch(fake_image))
            reward = reward - 0.5
            # reward = reward if self.goal == 0 else 1 - reward
            self.ind += 1

            if len(self.fake) == 1000:
                print('saving...')
                path = "./pics1/%d-%d.jpg" % (self.ind, int(self.goal))
                Image.fromarray((cropped_img*255).astype(np.uint8)).save(path)
                self.randomTrain()
                self.fake = []
                print('done!')
            if self.goal == 0:
                self.reward_average.append(reward)
            if len(self.reward_average) == 10:
                message = '\n RL Avg: (r/10): {:+.3f} Iter: {:+.3f}\n ---------------'
                avg = np.average(self.reward_average)
                message = message.format(avg, self.ind)
                self.reward_average = []
                print(message)
            #self.goal = 1 if self.goal == 0 else 0
            mask = utils.compute_box_mask(self.observe.shape, *self.choices)
            self.observe[np.logical_not(mask)] = 0
            # self.observe = paster(cropped_img)
        return self.observe, reward, terminal, {}
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
            utils.crop_array(self.rl_image_unflat, self.x_val_0,
                             self.x_val_1, self.y_val_0, self.y_val_1)
            pic, cropped = cropper(
                self.rl_image_unflat, self.x_val_0, self.x_val_1, self.y_val_0, self.y_val_1)
            reward = -0.01
            if not(type(cropped) == int) and not(type(cropped) == float):
                fake_image = np.expand_dims(cropped.copy(), axis=0)
                self.fake.append(np.squeeze(fake_image))
                reward = float(self.discriminator.predict_on_batch(fake_image))
                reward = reward if goal == 0 else 1 - reward
                self.goal = 1 if self.goal == 0 else 0

                self.ind += 1

                if len(self.fake) == 1000:
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
        self.choices = []
        self.disc_image = random.choice(self.test_images)
        self.observe = random.choice(self.train_images)
        self.rl_image_unflat = self.observe.copy()
        #if self.goal == 0:
        #    self.observe = np.append(
        #        self.observe, np.zeros(self.observe.shape[:2] + (1,)), axis=-1)
        #else:
        #    self.observe = np.append(
        #        self.observe, np.ones(self.observe.shape[:2] + (1,)), axis=-1)
        self.x_val = -1
        self.y_val = -1
        self.chooser = 0
        self.timestep = 0
        self.last_pred = 0
        #if self.ind >= 400:
        #    self.ind = 0
        return self.observe

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass

    def randomTrain(self, n_epochs=1):
        # Pre-train the discriminator model
        for epoch in range(n_epochs):
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
            # idx = np.random.randint(len(self.test_images), size=len(trainNp))
            testNp = self.test_images[idx]
            # testNp = self.test_images.copy()
            x_train = np.row_stack((testNp, trainNp))
            y_labels = [1]*len(testNp) + [0]*len(trainNp)
            self.discriminator.fit(x=x_train, y=y_labels,
                                   epochs=1, verbose=1, shuffle=True)
            evals = self.discriminator.evaluate(self.test_images_evaluate, y=[
                1]*len(self.test_images_evaluate))
            print(evals)
            evals = self.discriminator.evaluate(trainNp, y=[0]*len(trainNp))
            print(evals)
