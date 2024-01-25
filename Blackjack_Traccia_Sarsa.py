#!/usr/bin/env python
# coding: utf-8

# # Blackjack game with Q-Learning
# 
# ## Blackjack game
# 
# The goal of the game is to beat the dealer by having a hand value of 21 or closest to 21 without going over. Each player is dealt two cards, and then has the option to "hit" and receive additional cards to improve their hand. They can also choose to “stand” and keep the cards they have. If the player exceeds 21, they “bust” and automatically lose the round. If the player has exactly 21, they automatically win. Otherwise, the player wins if they are closer to 21 than the dealer. \
# The value of each card is listed below:
# - 10/Jack/Queen/King → 10
# - 2 through 9 → Same value as card
# - Ace → 1 or 11 (Player’s choice)
# 
# The game starts with players making their bets, after which the dealer will deal two cards to each player and two to himself, with one card face up and one face down (known as the "hole" card). Players then make their decisions to hit or stand. Once all players have completed their turns, the dealer will reveal their hole card and hit or stand according to a set of rules. If the dealer busts (goes over 21), players who have not bust win. If neither the player nor dealer busts, the hand closest to 21 wins.

# ## Importing modules
# 
# The modules used here are to call the openAI environment, with `gym`, and the basic plotting scheme with `matplotlib` module. `random` is used as the main part regarding the cards dealing and other necessary data management.\
# 

# In[1]:


from abc import ABC, abstractmethod
"""
"""
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import enum


# ## Setting up the basics of the game
# 
# ### Card and deck
# 
# The cards' definitions are set up below. 
# - `ranks`: a dictionary that maps the string representations of card ranks to their numerical values
# - `Suit`: an enumeration of the four suits in a standard deck of playing cards
# - `Card`: a class that represents an individual card and contains properties such as the card's suit, rank, and value
# - `Deck`: a class that represents a deck of cards and contains methods for shuffling the deck, dealing cards, peeking at the top card, adding cards to the bottom of the deck, and printing the deck
# 
# The `Deck` class initializes an empty list of cards when created and populates it with `num` standard decks of playing cards. The `shuffle` method randomly shuffles the cards in the deck. The `deal` method returns the top card of the deck, and the `peek` method returns the top card without removing it from the deck. The add_to_bottom method adds a card to the bottom of the deck, and the `__str__` method returns a string representation of the deck. The `__len__` method returns the number of cards in the deck.

# In[2]:


ranks = {
    "two" : 2,
    "three" : 3,
    "four" : 4,
    "five" : 5,
    "six" : 6,
    "seven" : 7,
    "eight" : 8,
    "nine" : 9,
    "ten" : 10,
    "jack" : 10,
    "queen" : 10,
    "king" : 10,
    "ace" : (1, 11)
    }

class Suit(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    diamonds = "diamonds"
    hearts = "hearts"
    
class Card:
    def __init__(self, suit, rank, value):
        self.suit = suit
        self.rank = rank
        self.value = value
        
    def __str__(self):
        return self.rank + " of " + self.suit.value
    
class Deck:
    def __init__(self, num=1):
        self.cards = []
        for i in range(num):
            for suit in Suit:
                for rank, value in ranks.items():
                    self.cards.append(Card(suit, rank, value))
                
    def shuffle(self):
        random.shuffle(self.cards)
        
    def deal(self):
        return self.cards.pop(0)
    
    def peek(self):
        if len(self.cards) > 0:
            return self.cards[0]
        
    def add_to_bottom(self, card):
        self.cards.append(card)
        
    def __str__(self):
        result = ""
        for card in self.cards:
            result += str(card) + "\n"
        return result
    
    def __len__(self):
        return len(self.cards)


# ### Setting up the game's rules
# 
# The evaluation of the dealer's hand is done here, following a proper set of rules that are predictable. As the game goes, the dealer will be following *Hard 17* rule. This means the dealer will not hit again if the Ace yields a 17. This also means that Aces initially declared as 11's can be changed to 1's as new cards come.

# In[3]:


def dealer_eval(player_hand):
    num_ace = 0
    use_one = 0
    for card in player_hand:
        if card.rank == "ace":
            num_ace += 1
            use_one += card.value[0] # It will be using Ace is equal to 1 in here
        else:
            use_one += card.value
    
    if num_ace > 0:
        ace_counter = 0
        while ace_counter < num_ace:
            use_eleven = use_one + 10 
            
            if use_eleven > 21:
                return use_one
            elif use_eleven >= 17 and use_eleven <= 21:
                return use_eleven
            else:
                use_one = use_eleven
            
            ace_counter += 1
        
        return use_one
    else:
        return use_one


# As for the evaluation of the player's hand, it focus on the value of Ace. For instance, if the Ace obtained gets the player to the total sum of 18 to 21, then Ace is equal to 11, otherwise it is equal to 1.

# In[4]:


def player_eval(player_hand):
    num_ace = 0
    use_one = 0
    for card in player_hand:
        if card.rank == "ace":
            num_ace += 1
            use_one += card.value[0] # It will be using Ace is equal to 1 in here
        else:
            use_one += card.value
    
    if num_ace > 0:
        ace_counter = 0
        while ace_counter < num_ace:
            use_eleven = use_one + 10 
            
            if use_eleven > 21:
                return use_one
            elif use_eleven >= 18 and use_eleven <= 21:
                return use_eleven
            else:
                use_one = use_eleven
            
            ace_counter += 1
        
        return use_one
    else:
        return use_one


# Below, the code for the logic in which the dealer will follow is presented. As the action is rather straightforward, nothing much happens. Following the Hard 17 rule, the dealer will stop hitting after the total is 17 or more.

# In[5]:


def dealer_turn(dealer_hand, deck):
    dealer_value = dealer_eval(dealer_hand)
    while dealer_value < 17:
        dealer_hand.append(deck.deal()) # Making the Hit
        dealer_value = dealer_eval(dealer_hand)

    return dealer_value, dealer_hand, deck


# ## Setting up the OpenAI environment for Blackjack
# 
# The solution is to create a custom class in order to train the bot.
# 
# The class has two main attributes:
# - the `action_space` attribute, which is a 2-element discrete space representing the two possible actions the player can take: hit (0) or stand (1)
# - the `observation_space` attribute, which is a tuple representing the state of the game, consisting of two elements: player hand value (18 possible values ranging from 3 to 20) and the dealer's upcard value (10 possible values ranging from 2 to 11).
# 
# The class also implements four main methods:
# 
# - `step(action)` method, which takes the player's action as an input, updates the game state, calculates the reward, and returns the new state and the reward.
# - `reset()` method, which resets the game to its initial state and returns the start state.
# - `_take_action(action)` method, which updates the player's hand according to the player's action.
# - `dealer_turn(dealer_hand, bj_deck)` method, which calculates the dealer's final hand value according to the dealer's rules.
# 
# The game starts with an initial balance of 1000, and a deck of cards made up of 6 decks. The deck is initialized using the `Deck` class. The game can end in three ways: the player stands, the player's hand value exceeds 21, or the dealer's hand value exceeds 21. The rewards are -1 for losing, 0 for tie, and 1 for winning.

# In[6]:


INITIAL_BALANCE = 1000
NUM_DECKS = 6

class BlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(BlackjackEnv, self).__init__()
        
        # Initialize the blackjack deck.
        self.bj_deck = Deck(NUM_DECKS)
        
        self.player_hand = []
        self.dealer_hand = []
        
        self.reward_options = {"lose":-1, "tie":0, "win":1}
        
        # hit = 0, stand = 1
        self.action_space = spaces.Discrete(2)
        
        # Second element of the tuple is the range of possible values for the dealer's upcard. (2 through 11)
        self.observation_space = spaces.Tuple((spaces.Discrete(18), spaces.Discrete(10)))
        
        self.done = False
        
    def _take_action(self, action):
        if action == 0: # hit
            self.player_hand.append(self.bj_deck.deal())
            
        # re-calculate the value of the player's hand after any changes to the hand.
        self.player_value = player_eval(self.player_hand)
    
    def step(self, action):
        self._take_action(action)
        
        # End the episode/game is the player stands or has a hand value >= 21.
        self.done = action == 1 or self.player_value >= 21
        
        # rewards are 0 when the player hits and is still below 21, and they keep playing.
        rewards = 0
        
        if self.done:
            # CALCULATE REWARDS
            if self.player_value > 21: # above 21, player loses automatically.
                rewards = self.reward_options["lose"]
            elif self.player_value == 21: # blackjack! Player wins automatically.
                rewards = self.reward_options["win"]
            else:
                ## Begin dealer turn phase.

                dealer_value, self.dealer_hand, self.bj_deck = dealer_turn(self.dealer_hand, self.bj_deck)

                ## End of dealer turn phase

                #------------------------------------------------------------#

                ## Final Compare

                if dealer_value > 21: # dealer above 21, player wins automatically
                    rewards = self.reward_options["win"]
                elif dealer_value == 21: # dealer has blackjack, player loses automatically
                    rewards = self.reward_options["lose"]
                else: # dealer and player have values less than 21.
                    if self.player_value > dealer_value: # player closer to 21, player wins.
                        rewards = self.reward_options["win"]
                    elif self.player_value < dealer_value: # dealer closer to 21, dealer wins.
                        rewards = self.reward_options["lose"]
                    else:
                        rewards = self.reward_options["tie"]
        
        self.balance += rewards
        
        
        # Subtract by 1 to fit into the possible observation range.
        # This makes the possible range of 3 through 20 into 1 through 18
        player_value_obs = self.player_value - 2
        
        # get the value of the dealer's upcard, this value is what the agent sees.
        # Subtract by 1 to fit the possible observation range of 1 to 10.
        upcard_value_obs = dealer_eval([self.dealer_upcard]) - 1
        
        # the state is represented as a player hand-value + dealer upcard pair.
        obs = np.array([player_value_obs, upcard_value_obs])
        
        return obs, rewards, self.done, {}
    
    def reset(self): # resets game to an initial state
        # Add the player and dealer cards back into the deck.
        self.bj_deck.cards += self.player_hand + self.dealer_hand

        # Shuffle before beginning. Only shuffle once before the start of each game.
        self.bj_deck.shuffle()
         
        self.balance = INITIAL_BALANCE
        
        self.done = False
        
        # returns the start state for the agent
        # deal 2 cards to the agent and the dealer
        self.player_hand = [self.bj_deck.deal(), self.bj_deck.deal()]
        self.dealer_hand = [self.bj_deck.deal(), self.bj_deck.deal()]
        self.dealer_upcard = self.dealer_hand[0]
        
        # calculate the value of the agent's hand
        self.player_value = player_eval(self.player_hand)
        
        # Subtract by 1 to fit into the possible observation range.
        # This makes the possible range of 2 through 20 into 1 through 18
        player_value_obs = self.player_value - 2
            
        # get the value of the dealer's upcard, this value is what the agent sees.
        # Subtract by 1 to fit the possible observation range of 1 to 10.
        upcard_value_obs = dealer_eval([self.dealer_upcard]) - 1
        
        # the state is represented as a player hand-value + dealer upcard pair.
        obs = np.array([player_value_obs, upcard_value_obs])
        
        return obs
    
    def render(self, mode='human', close=False):
        # convert the player hand into a format that is
        # easy to read and understand.
        hand_list = []
        for card in self.player_hand:
            hand_list.append(card.rank)
            
        # re-calculate the value of the dealer upcard.
        upcard_value = dealer_eval([self.dealer_upcard])
        
        print(f'Balance: {self.balance}')
        print(f'Player Hand: {hand_list}')
        print(f'Player Value: {self.player_value}')
        print(f'Dealer Upcard: {upcard_value}')
        print(f'Done: {self.done}')
        
        print()


# After the environment for Blackjack is created, it is necessary to create the Agent, or bot, that will learn how to play the game.
# 
# With the learning method of choice defined as the **SARSA**, a class `Agent` represents a reinforcement learning agent that can learn from its environment through trial and error. The agent has various attributes such as the environment it interacts with, its exploration factor `epsilon`, learning rate `alpha`, discount factor `gamma`, and the number of episodes it needs to train on `num_episodes_to_train`. The `Agent` class has several methods that allow it to learn from its interactions with the environment:
# - `update_parameters` method: This method updates the exploration factor epsilon and the learning rate alpha after each action the agent takes. The epsilon decreases over time and the agent becomes less explorative as it trains, so that it takes more exploitation-based decisions in the future.
# 
# - `create_Q_if_new_observation` method: If the agent encounters a new observation, it sets the initial values of the Q-values for each action to 0.0. The Q-values represent the expected long-term reward of taking a specific action in a specific observation.
# 
# - `get_maxQ` method: Given an observation, this method returns the maximum Q-value of all the actions that the agent can take.
# 
# - `choose_action` method: Based on the current observation, the agent uses this method to choose an action to take. If a random number is greater than the exploration factor epsilon, the agent chooses the action with the highest Q-value. Otherwise, it takes a random action.
# 
# - `learn` method: This method updates the Q-value of the action the agent took based on the reward it received and the utility of the next observation. The new Q-value is computed using a formula that combines the current Q-value with the reward and the discounted utility of the next observation.

# In[114]:


class Agent:
    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.8, num_episodes_to_train=100000): #1.0, 0.1, 0.8
        self.env = env
        self.epsilon = epsilon  # Exploration factor
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.num_episodes_to_train = num_episodes_to_train
        self.Q = {}  # Q-values table

    def update_parameters(self, episode):
        # Decrease exploration factor epsilon over time
        self.epsilon = max(0.1, min(1.0, 1.0 - np.log10((episode + 1) / 25)))
        
        # Update learning rate alpha
        self.alpha = max(0.1, min(1.0, 1.0 - np.log10((episode + 1) / 25)))

    def create_Q_if_new_observation(self, observation):
        if observation not in self.Q:
            self.Q[observation] = [0.0, 0.0]  # Initialize Q-values to 0 for both actions

    def get_maxQ(self, observation):
        self.create_Q_if_new_observation(observation)
        return max(self.Q[observation])

    def choose_action(self, observation):
        self.create_Q_if_new_observation(observation)

        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Exploration: choose a random action
        else:
            maxQ = self.get_maxQ(observation)
            return np.argmax(self.Q[observation])  # Exploitation: choose the action with the highest Q-value

    def learn(self, state, action, reward, next_state, next_action):
        self.create_Q_if_new_observation(next_state)

        # SARSA update rule
        self.Q[state][action] = self.Q[state][action] + self.alpha * (
                reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

    def train(self):
        total_win = 0
        avg_win = 0
        list_avg = []
        count = 0
        for episode in range(self.num_episodes_to_train):
            state = self.env.reset()
            state = tuple(state)

            action = self.choose_action(state)

            total_reward = 0
            count += 1
            
            while True:
                next_state, reward, done, _ = self.env.step(action)
                next_state = tuple(next_state)
                next_action = self.choose_action(next_state)
                
                total_reward += reward
                
                if reward == 1:
                    total_win += reward
                
                # Learn from the experience
                self.learn(state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
            
                if done:
                    break
            # Update exploration and learning rates
            self.update_parameters(episode)
            
            #if episode % 10 == 0:
                
                # Calculate the winning average
                #avg_win = total_win / self.num_episodes_to_train
                #print(f"Episode: {episode}, Win Rate: {avg_win * 100}")
                
            # Calculate the winning average
            avg_win = total_win / count
            list_avg.append(avg_win)
            print(f"Episode: {episode}, Win Rate: {avg_win * 100}")
            #avg_win = total_win / 100
            #list_avg.append(avg_win)
            #total_win = 0
                

                 
        plt.plot(list_avg)
        plt.xlabel('Episodes')
        plt.ylabel('Win rate')
        plt.show()
            # Print episode information
            #if episode % 100 == 0:
                #print(f"Episode: {episode}, Win Rate: {avg_win * 100}")

    def test(self, num_episodes=10):
        total_rewards = 0
        total_win = 0
        i = 0
        
        for i in range(num_episodes):
            state = self.env.reset()
            state = tuple(state)
            action = self.choose_action(state)
            
            while True:
                next_state, reward, done, _ = self.env.step(action)
                next_state = tuple(next_state)
                action = self.choose_action(next_state)

                total_rewards += reward
                if reward == 1:
                    total_win += reward
                    
                if done:
                    break
            
        average_reward = total_rewards / num_episodes
        avg_win = total_win / num_episodes
        print(f"Average Reward over {num_episodes} episodes: {avg_win * 100}")


# In[115]:


#print(f"WE: {sarsa_agent.Q}")


# In[116]:


# Create Blackjack environment
env = BlackjackEnv()


# In[117]:


# Create SARSA agent
sarsa_agent = Agent(env)


# In[96]:


# Train the agent
sarsa_agent.train()


# In[98]:


print(sarsa_agent.epsilon)
print(sarsa_agent.alpha)


# In[125]:


# Test the agent
sarsa_agent.test()


# In[102]:


print(f"table: {sarsa_agent.Q}")


# ## Training the model
# 
# This snippet is a simulation of the reinforcement learning agent playing Blackjack. 
# 
# The agent is using a custom environment called `BlackjackEnv` and an `Agent` class, both previously defined here. The agent is trained over a set number of episodes (30000), where it takes actions based on a Q-table and updates the Q-table based on the rewards received from each action. The simulation is run for 1000 rounds with 1000 samples to calculate the average payout per round. The agent's average payout is then plotted and printed, with the running average present for easier comprehension.

# In[ ]:


# Loading the custom model
env = BlackjackEnv()

agent = Agent(env, epsilon=1.0, alpha=0.01, gamma=0.01, num_episodes_to_train=30000)

num_rounds = 1000 # Payout calculated over num_rounds
num_samples = 1000 # num_rounds simulated over num_samples

average_payouts = []

observation = env.reset()
for sample in range(num_samples):
    round = 1
    total_payout = 0 # to store total payout over 'num_rounds'
    # Take action based on Q-table of the agent and learn based on that until 'num_episodes_to_train' = 0
    while round <= num_rounds:
        action = agent.choose_action(observation)
        next_observation, payout, is_done, _ = env.step(action)
        agent.learn(observation, action, payout, next_observation)
        total_payout += payout
        observation = next_observation
        if is_done:
            observation = env.reset() # Environment deals new cards to player and dealer
            round += 1
    average_payouts.append(total_payout)
    
avg_run = moving_average(average_payouts)

# Plot payout per 1000 episodes for episode
plt.figure(figsize=(10,6))
plt.plot(average_payouts, 'k-')
plt.plot(avg_run, 'r-')
plt.xlabel('Number of samples')
plt.ylabel('Payout after {} rounds'.format(num_rounds))
plt.grid(linestyle=':')
plt.show()      
    
print ("Average payout after {} rounds is: {}".format(num_rounds, sum(average_payouts)/(num_samples)))


# In[ ]:




