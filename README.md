# Reinforcement Learning Project

Reinforcement learning project of ***Artificial Intelligence for Robotics 1*** of the master's course in *Robotics Engineering* at University of Genoa, curated by **Alessandro Trovatello**.

The aim of the project was to use reinforcement learning to implement an Agent able to improve its skills in playing blackjack, using the **SARSA** method. The template was provided by the professor and is a custom OpenAI's Blackjack environment without the Agent. The Agent is fully created from scratch using the *State - Action - Reward - nextState - nextAction* method.

## Blackjack Game

The goal of the game is to beat the dealer by having a hand value of 21 or the closest to 21 without going over. Each player is dealt two cards, then has the option to "hit" and receive additional cards to improve their hand or "stand" and keep their current cards. If the player exceeds 21, they "bust" and automatically lose the round. If the player has exactly 21, they automatically win. Otherwise, the player wins if they are closer to 21 than the dealer.

### Card Values:
- 10/Jack/Queen/King → 10
- 2 through 9 → Same value as card
- Ace → 1 or 11 (Player’s choice)

The game starts with players making their bets. The dealer then deals two cards to each player and two to himself, with one card face up and one face down (the "hole" card). Players then decide to hit or stand. Once all players have completed their turns, the dealer reveals their hole card and acts according to predefined rules. If the dealer busts, players who haven't busted win. Otherwise, the highest hand wins.

## Importing Modules

The following modules are used:
- `gym` for OpenAI's environment
- `matplotlib` for plotting
- `random` for handling card dealing and data management

## Setting Up the Basics of the Game

### Card and Deck Definitions

- `ranks`: Dictionary mapping card ranks to numerical values.
- `Suit`: Enumeration of the four suits in a standard deck.
- `Card`: Class representing an individual card (suit, rank, value).
- `Deck`: Class for handling deck operations such as shuffling and dealing cards.

The `Deck` class initializes an empty list of cards and populates it with `num` standard decks. Key methods include:
- `shuffle()`: Randomly shuffles the deck.
- `deal()`: Returns the top card.
- `peek()`: Returns the top card without removing it.
- `add_to_bottom()`: Adds a card to the bottom.

### Game Rules Implementation

The dealer follows the *Hard 17* rule, meaning:
- If the dealer’s Ace yields a 17, they will not hit again.
- Aces initially valued as 11 can be changed to 1 when necessary.

For the player's hand, the Ace's value is determined dynamically:
- If Ace makes the total 18-21 → it is worth 11.
- Otherwise, it is worth 1.

## OpenAI Environment for Blackjack

A custom class is created to train and test the agent.

### Key Attributes:
- `action_space`: Two possible actions: hit (0) or stand (1).
- `observation_space`: Tuple representing game state → (player's hand value, dealer's upcard value).

### Main Methods:
- `step(action)`: Updates game state and calculates reward.
- `reset()`: Resets the game.
- `_take_action(action)`: Updates player’s hand.
- `dealer_turn(dealer_hand, bj_deck)`: Implements dealer's logic.
- `render(game_num, action)`: Prints game status.

The game starts with an initial balance of 1000 and a deck of 6 standard decks. The agent earns:
- `-1` for losing
- `0` for a tie
- `1` for winning

## Implementing the SARSA Learning Algorithm

The `Agent` class is responsible for learning through trial and error.

### Key Methods:
- `update_parameters()`: Updates exploration factor `epsilon`.
- `create_Q_if_new_state()`: Initializes Q-values for new states.
- `choose_action()`: Selects an action based on Q-values or exploration.
- `learn()`: Updates Q-values using the SARSA update formula.
- `moving_average()`: Computes a moving average of results.

### SARSA Update Formula:
```math
Q(state, action) = Q(state, action) + alpha \cdot [reward + gamma \cdot Q(next\_state, next\_action) - Q(state, action)]
```
Where:
- `Q(state, action)`: Current Q-value
- `alpha`: Learning rate
- `reward`: Immediate reward
- `gamma`: Discount factor
- `Q(next_state, next_action)`: Estimated Q-value of the next state-action pair

### Training and Testing the Agent

Two key functions are implemented:
- `train()`: Trains the agent.
- `test()`: Tests the agent using the trained Q-table.

### Running the Agent

```python
# Create Blackjack environment
env = BlackjackEnv()

# Create SARSA agent
sarsa_agent = Agent(env, epsilon=1.0, alpha=0.1, gamma=0.01, num_episodes_to_train=10000)

# Train the agent
sarsa_agent.train()

# Test the agent
sarsa_agent.test(num_episodes_to_test=10000)
```

### Simulating the Agent's Performance

After training and testing, a simulation is run for `1000` rounds with `1000` samples to calculate the average payout per round, starting from an `INITIAL_BALANCE` of 1000. The agent's average payout is then plotted, showing how the agent learns over time.

![image](https://github.com/user-attachments/assets/bbb21df3-6256-4810-a096-3e6b10858a08)

Average payout after 1000 rounds is: 762.98

### Key Insights:
- The agent improves over approximately 100-150 samples.
- The `epsilon` value is updated at each step.
- Performance depends on hyperparameters (`alpha`, `gamma`).
- Higher `alpha` can lead to steeper learning but worse results.

To further confirm results, test the agent with 1000 games:
```python
agent.test(num_episodes_to_test=1000)
```
The result is:

- Balance: From initial 1000 to 946
- Win Rate:  44.3 %
- Lose Rate:  49.7 %
- Tie Rate:  6.0 %

This concludes the implementation of the Reinforcement Learning Blackjack Agent using SARSA.

