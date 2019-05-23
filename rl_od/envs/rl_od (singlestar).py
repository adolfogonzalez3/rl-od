from PIL import Image, ImageDraw
import random, os, time
import math
import numpy as np
import numpy.random as npr
import gym
from gym import spaces
from keras.models import Model, Sequential
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, InputLayer
import tensorflow as tf
import keras.backend as K
from randPic import *
from PygameVisualizer import PygameVisualizer
import math

def neuralNet():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    
    
    model = Sequential()
    model.add(InputLayer((44, 44, 1)))
    #input_shape = (2, 44, 44)
    #model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
     #                activation='sigmoid'))
                     #input_shape=input_shape))
    #model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

class rl_od(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.discriminator = neuralNet()
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.discriminator.compile(optimizer=tf.train.AdamOptimizer(0.1), loss='binary_crossentropy')
        action_size = 2
        observe_size = (150 ** 2)*3
        self.ind = 0
        self.chooser = False
        self.x_val = -1
        self.y_val = -1
        self.visual = PygameVisualizer()
        self.observe = 0
        self.observe_unflat = 0
        self.real_image = np.expand_dims(np.array(Image.open('real_image.png', 'r')), axis=0) // 255
        self.real_image = np.subtract(self.real_image, 1)       
        self.real_image[np.where(self.real_image > 0)] = 1
        self.got_it = True
        self.fake_image = -1
        self.y_label = [1, 0]
        self.x_data = 0
        self.action_space = spaces.Discrete(150)
        self.train = False
        self.first = 1
        self.observation_space = spaces.Box(0, 1, shape=(observe_size,),
                                            dtype=np.float)

    def step(self, action):
        did_fail = False
        info = {}
        if self.chooser == False:
            self.chooser = True
            terminal = False
            reward = 0.01
            self.x_val = action
        else:
            terminal = True
            self.y_val = action
            cropped = cropImage(self.observe_unflat, self.x_val, self.y_val)
            reward = 0
            if not(type(cropped) == int):
                self.fake_image = np.expand_dims(cropped.copy(), axis=0) / 255
                self.fake_image[np.where(self.fake_image == 0)] = 1
                self.fake_image[np.where(self.fake_image != 1)] = 0

                #self.fake_image[np.where(self.fake_image == 0)] = 1
                #self.fake_image[np.where(self.fake_image != 1)] = 0

                self.x_data = np.concatenate((self.real_image, self.fake_image))
                self.x_data = np.expand_dims(self.x_data, axis=-1)
                reward = self.discriminator.predict(self.fake_image[..., None])[0][0]
                self.train = True
                self.observe = np.array(drawRect(self.observe_unflat, self.x_val, self.y_val, self.ind, False)).flatten()
                self.ind += 1
                if reward >= .8:
                    np.array(drawRect(self.observe_unflat, self.x_val, self.y_val, self.ind, True)).flatten()
                    self.got_it == True
                    self.discriminator.train_on_batch(self.x_data, self.y_label)

                
                
                if self.ind < 2:
                    self.discriminator.train_on_batch(self.x_data, self.y_label)


                '''
                board_bigger = board.copy()
                rgb = np.zeros((200, 200, 3), dtype=np.uint8)
                rgb[...] = board_bigger.reshape((200, 200, 1)).astype(np.uint8)
                self.visual.blit(rgb)
                '''

                if self.ind % 10 == 0:
                    print('Real: ', np.sum(self.real_image))
                    print('Fake: ', np.sum(self.fake_image))
                    message = '\n Reward: {:+.3f}\n ---------------'
                    message = message.format(reward)
                    print(message)
        return self.observe, reward, terminal, info

    def reset(self):
        if self.got_it == True:
            self.observe = randomImage()
            self.observe_unflat = self.observe
            self.observe = np.array(self.observe).flatten() / 255
            self.got_it = False
        self.first = 0
        self.chooser = False
        self.fake_image = -1
        self.x_val = -1
        self.y_val = -1
        return self.observe

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass
