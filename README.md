# DQN training code for OpenAI Pong

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller - DeepMind Technologies - ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## Dependencies

- [theano](http://deeplearning.net/software/theano/)
- [lasagne](http://lasagne.readthedocs.io/en/latest/)
- [opencv](http://opencv.org/), used for screen preprocessing only.

## Train

Comes better with GPU. You have to train it 1k-2k of episodes (around 1.5kk-2.5kk of iterations) to get some reasonable behavior of the agent.

```bash
python train.py 
```

## Play
```bash
python play.py -m model.npz
```
You can get a simple (2000 episodes) pretrained model [here](https://drive.google.com/file/d/0B5A-XismrZrJa0tWQUlLVjhIdjQ/view?usp=sharin://drive.google.com/file/d/0B5A-XismrZrJa0tWQUlLVjhIdjQ/view?usp=sharing).

