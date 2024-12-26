import numpy as np

class TicTacToeEnv:
    def __init__(self, episodes=5, realPlayer=False):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lr = 0.1
        self.episodes = episodes

        self.board = np.array([
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ])

        self.actions = np.arange(9)  # 9 possible actions for a 3x3 board
        self.q_table = np.zeros((3**9, len(self.actions)))  # Q-table for state-action values
        self.realPlayer = realPlayer

    def board_to_state(self):
        """Convert the board configuration to a unique integer state"""
        state = 0
        for i, val in enumerate(self.board.flatten()):
            state += val * (3 ** i)
        return state

    def render(self):
        for r in range(self.board.shape[0]):
            for c in range(self.board.shape[1]):
                if self.board[r, c] == 1:
                    print("| X |", end=" ")
                elif self.board[r, c] == 2:
                    print("| O |", end=" ")
                else:
                    print("|   |", end=" ")
            print("\n")
        print("----------------------")

    def reset(self):
        """Reset the game board to its initial state"""
        self.board = np.array([
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ])

    def step(self, action):
        """Perform one action in the environment"""
        row, col = action // 3, action % 3

        if self.board[row, col] != 0:
            return -1  # Invalid move
        
        self.board[row, col] = 1  # Player (X) move
        
        if self.checkIfWin():
            return 100  # Player (X) wins

        self.opponent_move()

        if self.checkIfWin():
            return -100  # Opponent (O) wins

        return 0  # No win yet

    def updateQTable(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning formula"""
        next_max = np.max(self.q_table[next_state])  # Max Q-value of the next state
        self.q_table[state, action] += self.lr * (reward + (self.gamma * next_max) - self.q_table[state, action])

    def checkIfWin(self):
        """Check if the current player has won"""
        for x in range(3):
            if self.board[x][0] == self.board[x][1] == self.board[x][2] != 0:
                return True
            elif self.board[0][x] == self.board[1][x] == self.board[2][x] != 0:
                return True

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return True

        return False

    def opponent_move(self):
        """Let the opponent (O) make a move"""
        while True:
            random_action = np.random.choice(self.actions)
            rand_row, rand_col = random_action // 3, random_action % 3
            if self.board[rand_row, rand_col] == 0:
                self.board[rand_row, rand_col] = 2
                break

    def train_ai(self):
        """Train the AI using Q-learning"""
        for episode in range(self.episodes):
            self.reset()
            state = self.board_to_state()
            done = False
            total_reward = 0

            while not done:
                # Select action using epsilon-greedy policy
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.actions)
                else:
                    action = np.argmax(self.q_table[state])  # Choose the best action
                
                reward = self.step(action)
                next_state = self.board_to_state()

                # Update Q-table
                self.updateQTable(state, action, reward, next_state)

                # If there's a win, terminate the episode
                if reward == 100 or reward == -100:
                    done = True
                    total_reward += reward
                state = next_state

            print(f"Episode {episode+1} ended with reward: {total_reward}")


    def greedy_policy(self, state):
        return np.argmax(self.q_table[state])

env = TicTacToeEnv(500)
env.train_ai()
