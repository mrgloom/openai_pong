# DQN training code for OpenAI Pong

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller - DeepMind Technologies - ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## Dependencies

- [theano](http://deeplearning.net/software/theano/)
- [lasagne](http://lasagne.readthedocs.io/en/latest/)
- [opencv](http://opencv.org/), used only for screen preprocessing.

## Train

```bash
python train.py 
```

## Play
```bash
python play.py -m model.npz
```
You can get a simple (2000 episodes) pretrained model [here](https://drive.google.com/file/d/0B5A-XismrZrJa0tWQUlLVjhIdjQ/view?usp=sharin://drive.google.com/file/d/0B5A-XismrZrJa0tWQUlLVjhIdjQ/view?usp=sharing).

