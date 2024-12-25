import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lr = 0.1
        self.episodes = 100000

        self.board = np.array([
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ])

        self.actions = np.arange(9)
        self.q_table = {}

    def step(self, action):
        row, col = action // 3, action % 3
        if self.board[row, col] != 0:
            return self.get_state(), -5, True  
        
        self.board[row, col] = 1

        if self.check_if_won():
            return self.get_state(), 100, True

        if np.all(self.board != 0):
            return self.get_state(), 0, True

        self.opponent_move()

        if self.check_if_won():
            return self.get_state(), -100, True

        return self.get_state(), 0, False

    def render(self):
        for row in self.board:
            print(" | ".join(["X" if cell == 1 else "O" if cell == 2 else " " for cell in row]))
            print("-" * 9)
        print("\n")

    def reset(self):
        self.board = np.array([
            [0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]
        ])

    def check_if_won(self):
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

    def check_if_done(self):
        pass

    def opponent_move(self):
        if self.episodes < 2000:
            while True:
                random_action = np.random.choice(self.actions)
                rand_row, rand_col = random_action // 3, random_action % 3
                if self.board[rand_row, rand_col] == 0:
                    self.board[rand_row, rand_col] = 2
                    break
        
        else:
            self.greedy_policy(self.get_state())

    def get_state(self):
        return tuple(self.board.flatten())

    def train_ai(self):
        initial_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.995

        for episode in range(self.episodes):
            self.reset()
            self.epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))
            state = self.get_state()
            done = False
            total_reward = 0

            while not done:
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.actions)
                else:
                    q_values = [self.q_table.get((state, a), 0) for a in self.actions]
                    action = np.argmax(q_values)
                
                next_state, reward, done = self.step(action)
                total_reward += reward

                next_max = max([self.q_table.get((next_state, a), 0) for a in self.actions])
                self.q_table[(state, action)] = self.q_table.get((state, action), 0) + \
                    self.lr * (reward + self.gamma * next_max - self.q_table.get((state, action), 0))

                state = next_state

            if episode%10000==0:
                print(f"Episode {episode+1} ended with reward: {total_reward}")



    def greedy_policy(self, state):
        valid_actions = [
            action for action in self.actions
            if self.board[action // 3, action % 3] == 0
        ]
        if not valid_actions:
            return None  # No valid moves left
        q_values = [self.q_table.get((state, action), 0) for action in valid_actions]
        return valid_actions[np.argmax(q_values)]


    def play_against_ai(self):
        while True:
            choice = int(input("Play(1) or exit(0): "))
            if choice == 0:
                break
            self.reset()
            done = False
            while not done:
                self.render()

                # Player's (human's) turn (Player 2)
                player_move = int(input("Enter your move (1-9): "))
                player_move -= 1
                row, col = player_move // 3, player_move % 3
                if self.board[row, col] != 0:
                    print("Invalid move. Try again.")
                    continue
                self.board[row, col] = 2

                if self.check_if_won():
                    self.render()
                    print("You win!")
                    break

                if np.all(self.board != 0):
                    self.render()
                    print("It's a draw!")
                    break

                ai_move = self.greedy_policy(self.get_state())
                if ai_move is None:
                    self.render()
                    print("It's a draw!")
                    break
                row, col = ai_move // 3, ai_move % 3
                self.board[row, col] = 1

                if self.check_if_won():
                    self.render()
                    print("AI wins!")
                    break

                if np.all(self.board != 0):
                    self.render()
                    print("It's a draw!")
                    break




env = TicTacToeEnv()
env.train_ai()
env.play_against_ai()
