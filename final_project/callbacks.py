import os
import pickle
import random

import numpy as np

import settings as s
cols, rows = s.COLS-2, s.ROWS-2 # inner tiles

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
    if self.train or not os.path.isfile("my-saved-model.csv"):
        self.logger.info("Setting up model from scratch.")
        self.training_mode = True
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        self.training_mode = False
        #self.model = file.read()
        #str_model = np.loadtxt("my-saved-model.txt", dtype=str)
        self.model = np.loadtxt('my-saved-model.csv', delimiter=',')
        #with open("my-saved-model.txt", "r") as file:
        #    self.model = file.read()
        #    print(self.model.dtype())
               
            
    # initialize Q(s,a)
    # s=0,1,2,3
    # a=0,1,2,3,4,5
    #qzero = np.random.rand(4,6)
    #qzero[-1,:] = 0 # terminal state
    #piZero = np.argmax(qzero, axis=1)
    #self.Q = [qzero]
    #self.Pi = [piZero]
    #self.outcome = []


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if not self.training_mode:
        state = state_to_features(game_state)
        action = self.model[state].astype(int)
        return ACTIONS[action]
    else:
        eps = 0.1
        
        state = state_to_features(game_state)
        
        #print(self.Pi)
        #print(self.Q)
        
        if np.random.rand() < eps:
            #print('random action')
            action = np.random.randint(4)
        else:
            #print('best action')
            #print(state)
            action = np.argmax(self.Q[state,:])
            #print(action)
        #print('Action: ', ACTIONS[action])
        return ACTIONS[action]
    
    '''
    #print(game_state['explosion_map'])
    # state s=0,1,2,3
    if not game_state['coins']:
        gstate=3
        print(game_state['coins'])
        print('NO COINS LEFT')
    else:
        coins = np.asarray(game_state['coins'][0])
        selfpos = np.asarray(game_state['self'][-1])
        gstate = np.sum(np.abs(selfpos-coins))
        
    # exploitation vs exploration
    eps = 0.7 # large initial epsilon
    
    if np.random.rand() < eps:
        action = np.random.randint(6)
    else:
        action = np.argmax(self.Q[-1][gstate,:])
        
    return ACTIONS[action]

    
    # take random action
    return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])
    
    
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
    '''

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array 5*17*17
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # Gather information about the state
    field = game_state['field']  # 1 crates, -1 stone walls, 0 free tiles
    # field_inside = field[1:cols+1, 1:rows+1]
    # print(field_inside.shape)
    # print(field)
    bombs = game_state['bombs']  #[(x, y), t]
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']  #[(x, y)]
    sx, sy = game_state['self'][3]
    opponents = game_state['others']
    # with open('myfile0.txt', 'w') as f:
    #     print(game_state['self'], file=f)

    # with open('myfile1.txt', 'w') as f:
    #     print(game_state['others'], file=f)
    # print(game_state['others'])

    # 1 free tiles, -1 stone walls, 0 crates
    channel_1 = field
    # print(channel_1)
    ones = channel_1 == 1
    zeros = channel_1 == 0
    channel_1[ones] = 0
    channel_1[zeros] = 1

    # print(channel_1)

    # Position contains the player
    channel_2 = np.zeros_like(field)
    channel_2[sx, sy] = 1
    # print(channel_2)

    # Position contains an opponent
    channel_3 = np.zeros_like(field)
    for opponent in opponents:
        channel_3[opponent[3]]= 1
    # print(channel_3)

    # Position contains a coin
    channel_4 = np.zeros_like(field)
    # print(coins)
    for coin in coins:
        channel_4[coin] = 1
    # print(channel_4)

    # Danger level of position (-1 <= danger <= 1)
    if bombs is None:
        channel_5 = np.zeros_like(field)
    else:
        channel_5 = explosion_map
        # print(bombs)
        channel_5 = np.where(explosion_map != 0, (s.BOMB_TIMER - explosion_map)/s.BOMB_TIMER, 0) #need to add cases when danger = 1

        for bomb in bombs:
            bx, by = bomb[0]
            distance = np.abs(bx - sx) + np.abs(by - sy)
            if distance == s.BOMB_TIMER - bomb[1]:
                danger_c = -1
            else:
                danger_c = 1

            #danger is negative if player self places it
            #danger is positive if opponents place it
            for i in range(1, s.BOMB_POWER + 1):
                if field[sx + i, sy] == -1:
                    break
                if bomb[1] <= 0: channel_5[sx + i, sy] = 1 #add cases when danger value = 1
                channel_5[sx + i, sy] *= danger_c
            for i in range(1, s.BOMB_POWER + 1):
                if field[sx - i, sy] == -1:
                    break
                if bomb[1] <= 0: channel_5[sx + i, sy] = 1
                channel_5[sx + i, sy] *= danger_c
            for i in range(1, s.BOMB_POWER + 1):
                if field[sx, sy + i] == -1:
                    break
                if bomb[1] <= 0: channel_5[sx + i, sy] = 1
                channel_5[sx + i, sy] *= danger_c
            for i in range(1, s.BOMB_POWER + 1):
                if field[sx, sy - i] == -1:
                    break
                if bomb[1] <= 0: channel_5[sx + i, sy] = 1
                channel_5[sx + i, sy] *= danger_c
    # print("======")
    # print(channel_5)
    channels = np.stack((channel_1.T, channel_2.T, channel_3.T, channel_4.T, channel_5.T))
    print((channels.reshape(-1)).shape)
    # return them as a vector
    return channels.reshape(-1)  # .reshape(-1)


    '''

    if not game_state['coins']:
        print('NO COINS LEFT, GAME FINISHED.')
        return (2*cols-1)*(2*rows-1) # = maximal state number when no coin left (terminal state)
    px, py = game_state['self'][3]
    minDist = cols+rows
    for c in game_state['coins']:
        cx, cy = c
        #print(cx, cy)
        cdx = cx-px
        cdy = cy-py
        #print(cdx, cdy)
        currentDist = abs(cdx)+abs(cdy)
        if currentDist < minDist:
            minDist = currentDist
            state = cdx+(cols-1) + (2*cols-1)*(cdy+(rows-1))
    return state
    '''

    '''
    # USING cx,cy,px,py
    if not game_state['coins']:
        print('NO COINS LEFT, GAME FINISHED.')
        state = (cols*rows)**2 # = maximal state number when no coin left (terminal state)
    else:
        #print(game_state['coins'][0])
        cx, cy = game_state['coins'][0]
        #print(cx, cy)
        cx -= 1
        cy -= 1
        #print(cx, cy)
        #print(game_state['self'][3])
        px, py = game_state['self'][3]
        #print('p:',px,py)
        px -= 1
        py -= 1
        #print(px,py)
        
        state_vector = np.stack((cx,cy,px,py))
        #print(state_vector)
        state = py+rows*px+(cols*rows)*cy+(cols*rows**2)*cx # binary code
    return state
    '''
    '''
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    channel_coins = np.zeros_like(game_state['field'])
    for c in game_state['coins']:
        channel_coins[c] = 1
    channel_players = np.zeros_like(game_state['field'])
    for p in game_state['others']:
        channel_players[p] = -1
    if game_state['self'][2] == True:
        channel_players[game_state['self'][3]] = 2 # if action BOMB available
    else:
        channel_players[game_state['self'][3]] = 1
    channel_bombs = np.zeros_like(game_state['field'])
    for b in game_state['bombs']:
        channel_bombs[b[0]] = b[1]
    #channels.append(game_state['field'], channel_players, channel_coins, game_state['explosion_map'])
        
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack((game_state['field'].T, channel_players.T, channel_coins.T, channel_bombs.T))
    # and return them as a vector
    return stacked_channels#.reshape(-1)
    '''
