'''
Batched version of the DQN algorithm with replay memory.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller - DeepMind Technologies - "Playing Atari with Deep Reinforcement Learning" https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

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

import gym

gamma = 0.99
batch_size= 32

game_limit = 10000  # max number of episodes to play
n_observe = 100000  # number of steps to take before decreasing the 'eps' 
n_explore = 1000000 # number of steps to decrease the 'eps'
memory_size = 50000 # replay memory size
eps_start = 1.
eps_end = 0.01
update_rate = 5000  # steps to perform before cloning the stale network

def create_dqn(n_acts):

    s = T.ftensor4('state')
    s_new = T.ftensor4('new_state')
    a = T.lvector('action')
    r = T.fvector('reward')
    f = T.lvector('finals')

    network = build_nature_with_pad3(n_acts, s)
    network_stale = build_nature_with_pad3(n_acts, s_new)

    s0 = lasagne.layers.get_output(network)
    s1 = lasagne.layers.get_output(network_stale)

    target = r + gamma*T.max(s1, axis=1)*(1 - f)

    loss = T.mean( T.sqr(target - s0[T.arange(batch_size),a] ))
    q_func = theano.function(inputs=[s], outputs=s0)

    params = lasagne.layers.get_all_params(network, trainable=True)
    params_stale = lasagne.layers.get_all_params(
            network_stale, trainable=True)

    updates = lasagne.updates.adadelta(loss, params, epsilon=1e-8)

    update_func = theano.function(inputs=[s, a, s_new, r, f],
        outputs=loss, updates=updates) 

    return q_func, update_func, network, network_stale

def clone_network(network, network_stale):
    lasagne.layers.set_all_param_values(
        network_stale, lasagne.layers.get_all_param_values(network))

def process_screen(screen):
    img = cv2.cvtColor(cv2.resize(screen, (80, 80)), 
                                    cv2.COLOR_RGB2GRAY)
    img = (img/255.).astype(np.float32)

    return img

def run(model, env, n_actions, eps_start):

    eps = eps_start
    eps_step = (eps_start-eps_end)/n_explore
    stack = deque()
    q_func, update_func, n0, n1 = model

    Q = []
    cost = []
    R = []
    update_steps = 0
    i = 0

    t0 = timer()

    for e in xrange(game_limit):

        done = False
        env_state = env.reset()

        state = np.zeros((4,80,80), dtype=np.float32)
        state[0] = process_screen(env_state)

        while (not done): 

            #print a_dist[0]
            # run an episode
            if np.random.uniform(0,1) > eps:
                a_dist = q_func([state])
                a = np.argmax(a_dist[0])        # playing single game
            else:
                # pick random action
                a = np.random.randint(0,n_actions)

            a = np.int8(a)
            
            env_state, reward, done, info = env.step(a)
            new_state = np.roll(state, 1, axis=0)
            new_state[0] = process_screen(env_state)

            # scale the reward a bit (game-specific, optional)
            r = np.float32(reward*10)

            # adjust 'final' flag type
            f = np.int8(done)

            R.append(r)
            Q.append(q_func([state])[0].mean())

            stack.append((state.copy(), a, new_state.copy(), r, f))
            state = new_state.copy()

            # making an update
            if len(stack) > batch_size and np.random.rand() > 0.5:
                batch = random.sample(stack, batch_size)
                s_ =        np.array([b[0] for b in batch])
                a_ =        np.array([b[1] for b in batch]).flatten()
                s_new_ =    np.array([b[2] for b in batch])
                r_ =        np.array([b[3] for b in batch]).flatten()
                f_ =        np.array([b[4] for b in batch]).flatten()

                loss = update_func(s_, a_, s_new_, r_, f_)
                cost.append(loss)
                update_steps += 1

            if len(stack) >= memory_size:
                for _ in range(batch_size):
                    stack.popleft()

            if update_steps > update_rate:
                clone_network(n0, n1)
                update_steps = 0

            if i and i % 1000 == 0:
                print 'Iteration: %d, Q: %.4f, eps: %.2f, cost: %.4f, R: %.4f, %.2f sec' % (i, np.mean(Q), eps, np.mean(cost), np.mean(R), timer() - t0)

                t0 = timer()
                Q = []
                R = []
                cost = []

            if i and (i % 10000) == 0:
                print "Saving model"
                np.savez('model.npz', 
                    *lasagne.layers.get_all_param_values(n0))

            if i > n_observe and eps > eps_end:
                eps -= eps_step

            i+=1

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=None, 
        help='Pre-trained model')
    parser.add_argument('--eps', '-e', type=float, default=1., 
        help='Initial epsilon')

    args = parser.parse_args()

    env = gym.make('Pong-v0')

    n_actions = env.action_space.n
    model = create_dqn(n_acts=n_actions) 

    if args.model:
        print "Loading network from:", args.model
        q_func, update_func, network, network_stale = model

        # load parameters and set up both networks
        with np.load(args.model) as h:
            param_values = [h['arr_%d' % i] 
                for i in range(len(h.files))]
            lasagne.layers.set_all_param_values(network, param_values)
            clone_network(network, network_stale)
           
    run(model, env, n_actions, eps_start=args.eps)

