import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            
    # initialize Q(s,a)
    # s=0,1,2,3
    # a=0,1,2,3,4,5
    qzero = np.random.rand(4,6)
    qzero[-1,:] = 0 # terminal state
    piZero = np.argmax(qzero, axis=1)
    self.Q = [qzero]
    self.Pi = [piZero]
    self.outcome = []


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # state s=0,1,2,3
    if not game_state['coins']:
        state=3
        print(game_state['coins'])
        print('NO COINS LEFT')
    else:
        coins = np.asarray(game_state['coins'][0])
        selfpos = np.asarray(game_state['self'][-1])
        state = np.sum(np.abs(selfpos-coins))
        
    # exploitation vs exploration
    eps = 0.7 # large initial epsilon
    
    if np.random.rand() < eps:
        action = np.random.randint(6)
    else:
        action = np.argmax(self.Q[-1][state,:])
        
    return ACTIONS[action]

    
    # take random action
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])
    
    """
    coins = game_state['coins'][0]
    selfpos = game_state['self'][-1]
    coin_dir = tuple(x-y for x,y in zip(coins,selfpos))
    print('\n', coins)
    print('\n', selfpos)
    print('\n', coin_dir)
    right = coin_dir[0] > 0
    left = coin_dir[0] < 0
    up = coin_dir[1] > 0
    down = coin_dir[1] < 0
    if game_state['field'][coins[0], coins[1]+1] == 0:
        if game_state['field'][selfpos[0]+1, selfpos[1]] == 0:
            if right: return('RIGHT')
            elif left: return('LEFT')
            elif up: return('UP')
            else: return('DOWN')
    
    if game_state['field'][selfpos[0], selfpos[1]+1] == 0:
        if up: return('UP')
        elif down: return('DOWN')
    return('BOMB')
        #return('RIGHT') if right else return('LEFT')
        #return('RIGHT')
   # if tile_is_free(self, selfpos[0], selfpos[1]+1) == 0:
    #    return('UP') if up else return('DOWN')
    
    
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)
    """

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
