import csv
import random


class MDPInitializer:
    """
    Class to generate state space.
    """

    def __init__(self, data_path, k, alpha):
        """
        The constructor for the MDPInitializer class.
        Parameters:
        :param data_path: path to data
        :param k: the number of items in each state
        :param alpha: the proportionality constant when considering transitions
        """

        self.u_path = data_path + "/users.csv"
        self.t_path = data_path + "/transactions.csv"
        self.g_path = data_path + "/games.csv"
        self.k = k
        self.alpha = alpha
        self.total_sequences = {}

        self.game_data = {}
        self.transactions = {}
        # Get user data and initialise transactions under each user
        self.fill_user_data()
        # Store transactions as { user_id : { game_title : [ play, purchase, play, ... ], ... }, ... }
        self.fill_transaction_data()

        self.actions, self.games, self.game_price = self.get_action_data()
        self.num_of_actions = len(self.actions)

    def fill_user_data(self):
        """
        The method to fill user data.
        :return: None
        """

        with open(self.u_path) as f:
            csv_f = csv.reader(f)
            next(csv_f)
            for row in csv_f:
                self.transactions[row[0]] = []

    def fill_transaction_data(self):
        """
        The method to fill the transactions for each user.
        :return: None
        """

        with open(self.t_path) as f:
            csv_f = csv.reader(f)
            next(csv_f)
            for row in csv_f:
                if row[1] not in self.transactions[row[0]]:
                    self.transactions[row[0]].append(row[1])
                if row[1] not in self.game_data:
                    self.game_data[row[1]] = [0, 0]
                self.game_data[row[1]][0] += float(row[3])
                self.game_data[row[1]][1] += 1

        for game in self.game_data:
            self.game_data[game] = self.game_data[game][0] / self.game_data[game][1]

    def get_action_data(self):
        """
        The method to obtain all games which will be actions.
        :return: list of the games/actions
        """

        actions = []
        games = {}
        game_price = {}
        with open(self.g_path) as f:
            csv_f = csv.reader(f)
            next(csv_f)
            for row in csv_f:
                actions.append(row[0])
                games[row[0]] = row[1]
                game_price[row[0]] = int(row[2])
        return actions, games, game_price

    def generate_initial_states(self):
        """
        The method to generate an initial state space.
        :return: states and the corresponding value vector
        """

        states = {}
        state_value = {}
        policy = {}
        policy_list = {}

        for user in self.transactions:
            # Prepend Nones for first transactions
            pre = []
            for i in range(self.k - 1):
                pre.append(None)
            games = pre + self.transactions[user]

            # Generate states of k items
            for i in range(0, len(games) - self.k + 1):
                temp_tup = ()
                for j in range(self.k):
                    temp_tup = temp_tup + (games[i + j],)

                if temp_tup in states:
                    states[temp_tup] = states[temp_tup] + 1
                else:
                    states[temp_tup] = 1
                    state_value[temp_tup] = 0
                    policy[temp_tup] = random.choice(self.actions)
                    policy_list[temp_tup] = random.sample(self.actions, len(self.actions))
                    for ind in range(len(policy_list[temp_tup])):
                        policy_list[temp_tup][ind] = (policy_list[temp_tup][ind], 1)

            # Generate states of k+1 items
            for i in range(0, len(games) - self.k - 1):
                temp_tup = ()
                for j in range(self.k + 1):
                    temp_tup = temp_tup + (games[i + j],)
                if temp_tup in self.total_sequences:
                    self.total_sequences[temp_tup] = self.total_sequences[temp_tup] + 1
                else:
                    self.total_sequences[temp_tup] = 1

        return states, state_value, policy, policy_list

    def generate_transitions(self, states, actions):
        """
        The method to generate the transition table.
        :param states: the initial states
        :param actions: the actions/items that can be chosen
        :return: a dictionary with transition probabilities
        """

        # Initialize the transitions dict
        transitions = {}

        # Store transitions as { state: { action/item chosen: { next_state: (alpha * count, reward), ... }, ... }, ... }
        for state, state_count in states.items():
            for action in actions:
                # Compute the new state
                new_state = ()
                for i in range(1, self.k):
                    new_state = new_state + (state[i],)
                new_state = new_state + (action,)

                # Compute the complete sequence
                total_sequence = state + (action,)
                # Find number of times the total sequence occurs
                if total_sequence not in self.total_sequences:
                    total_sequence_count = 1
                    self.total_sequences[total_sequence] = total_sequence_count
                else:
                    total_sequence_count = self.total_sequences[total_sequence]

                # Fill the transition probabilities
                if state not in transitions:
                    transitions[state] = {}
                if action not in transitions[state]:
                    transitions[state][action] = {}
                # Need to alpha * transition[state][action][n_state] as the action corresponds to the desired state
                transitions[state][action][new_state] = (self.alpha * total_sequence_count / state_count,
                                                         self.reward(new_state))

        # Adding the other possibilities and their probabilities for a particular action
        for state in transitions:
            for action in transitions[state]:
                for a in actions:
                    # Compute the new state
                    new_state = ()
                    for i in range(1, self.k):
                        new_state = new_state + (state[i],)
                    new_state = new_state + (a,)

                    # Need to beta * transition[state][a][n_state] as the action doesn't correspond to the desired state
                    if new_state not in transitions[state][action]:
                        transitions[state][action][new_state] = (self.beta(action, new_state)
                                                                 * transitions[state][a][new_state][0],
                                                                 self.reward(new_state))

        # Normalizing the probabilities
        for state in transitions:
            for action in transitions[state]:
                total = 0
                for new_state in transitions[state][action]:
                    total += transitions[state][action][new_state][0]
                for new_state in transitions[state][action]:
                    old_tup = transitions[state][action][new_state]
                    transitions[state][action][new_state] = (old_tup[0] / total, old_tup[1])

        return transitions

    def beta(self, action, new_state):
        """
        Method to calculate the beta required
        :param action: the action taken
        :param new_state: the new state
        :return: beta
        """

        # The difference in number of hours per unit currency
        diff = abs((self.game_data[action] / self.game_price[action]) -
                   (self.game_data[new_state[self.k - 1]] / self.game_price[new_state[self.k - 1]]))
        return diff / 120

    def reward(self, state):
        """
        Method to calculate the reward for each state
        :param state: the state
        :return: the reward for the given state
        """

        # spent = 0
        # for i in range(len(state) - 1):
        #     if state[i] is None:
        #         spent += 0
        #     else:
        #         spent += self.game_price[state[i]]
        # # The average amount spent before this purchase
        # if not len(state) == 1:
        #     spent /= (len(state) - 1)
        # y = spent / self.game_price[state[self.k - 1]]
        #
        # if y > 1:
        #     y = 1/y
        #
        # return (1 - y) * (self.game_data[state[self.k - 1]]) + y * (self.game_price[state[self.k - 1]])

        return 1
