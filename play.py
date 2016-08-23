'''
Batched version of the single q-learning algorithm
with convolutional model 

'''
from __future__ import division
import theano
from theano import tensor as T
from theano.misc import pkl_utils
import pylab as pl

from timeit import default_timer as timer

import random

import numpy as np
import cv2
import os
import cPickle as pickle
from collections import deque

import lasagne
from lasagne.layers import dnn, batch_norm

from scipy.misc import imresize
from models import build_nature_with_pad3
from train import create_dqn, process_screen

import gym

n_hid = 64
n_channels = 4

def run(q_func, env, eps):

    n_actions = env.action_space.n

    done = False
    env_state = env.reset()
    t0 = timer()

    state = np.zeros((4,80,80), dtype=np.float32)
    state[0] = process_screen(env_state)

    while (not done): 

        if np.random.rand() > eps:
            a_dist = q_func([state])
            a = np.argmax(a_dist[0])        # playing single game
            print a_dist[0]
        else:
            # pick random action
            a = np.random.randint(0,n_actions)

        env_state, r, done, info = env.step(a)
        new_state = np.roll(state, 1, axis=0)
        new_state[0] = process_screen(env_state)

        env.render()

        state = new_state.copy()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', required=True, 
        help='Pre-trained model')
    parser.add_argument('--eps', '-e', type=float, default=0., 
        help='Initial epsilon')

    args = parser.parse_args()

    env = gym.make('Pong-v0')

    n_actions = env.action_space.n
    model = create_dqn(n_acts=n_actions) 

    print "Loading network from:", args.model
    q_func, update_func, network, network_stale = model

    # load parameters and set up both networks
    with np.load(args.model) as h:
        param_values = [h['arr_%d' % i] 
            for i in range(len(h.files))]
        lasagne.layers.set_all_param_values(network, param_values)
       
    run(q_func, env, args.eps)

