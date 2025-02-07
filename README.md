# **TicTacToe-RL-CustomEnv**  
A custom Tic Tac Toe environment built using `gymnasium` to simulate and analyze gameplay for reinforcement learning applications. The environment supports player switching, move validation, and visualization of game states and outcomes.

## **Features**  
✅ **Custom RL Environment** – Implements a fully functional Tic Tac Toe environment using `gymnasium`.  
✅ **Game State Logging** – Tracks moves, rewards, and game progress for analysis.  
✅ **Board Visualization** – Displays board states as a table for better readability.  
✅ **Outcome Analysis** – Generates pie charts for game results (X wins, O wins, Draws).  
✅ **Invalid Move Handling** – Prevents illegal moves and ensures valid gameplay.  
✅ **Extensible for RL** – Designed for integration with RL algorithms like Q-learning.  

---

## **Installation**  
Ensure you have `gymnasium` installed:  
```bash
pip install gymnasium
```
Clone the repository:  
```bash
git clone https://github.com/yourusername/TicTacToe-RL-CustomEnv.git
cd TicTacToe-RL-CustomEnv
```

---

## **Usage**  
### **1. Import Required Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import Env, spaces
```

### **2. Define the Tic Tac Toe Environment**  
```python
class TicTacToeEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)  # 9 cells
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=int)
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X starts
        self.done = False
        return self.board, {}

    def step(self, action):
        if self.done:
            raise ValueError("Game over. Reset the environment.")

        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: Cell occupied.")

        self.board[row, col] = self.current_player
        reward, self.done = self.check_game_status()
        self.current_player *= -1
        return self.board, reward, self.done, {}

    def check_game_status(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return (1 if self.current_player == 1 else -1), True
        if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return (1 if self.current_player == 1 else -1), True
        if not np.any(self.board == 0):  # Draw
            return 0, True
        return 0, False

    def render(self, mode="human"):
        symbols = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join([symbols[cell] for cell in row]))
        print()
```

---

## **3. Visualization Functions**  
### **3.1 Board as Table**  
```python
def display_board_as_table(board):
    df = pd.DataFrame(board, columns=["Col 1", "Col 2", "Col 3"], index=["Row 1", "Row 2", "Row 3"])
    symbols = {1: "X", -1: "O", 0: "."}
    df = df.replace(symbols)
    print(df)
```

### **3.2 Game Logs**  
```python
def log_gameplay(env):
    logs = []
    state, _ = env.reset()
    done = False
    step = 0

    while not done:
        valid_moves = np.argwhere(state.flatten() == 0)
        action = np.random.choice(valid_moves.flatten())
        row, col = divmod(action, 3)
        state, reward, done, _ = env.step(action)

        logs.append({"Step": step + 1, "Player": "X" if env.current_player == -1 else "O",
                     "Action": f"({row}, {col})", "Reward": reward, "Done": done})
        display_board_as_table(state)
        step += 1

    return pd.DataFrame(logs)
```

### **3.3 Outcome Visualization**  
```python
def visualize_outcomes(results):
    outcomes = {"X Wins": results.count(1), "O Wins": results.count(-1), "Draws": results.count(0)}

    plt.figure(figsize=(6, 6))
    plt.pie(outcomes.values(), labels=outcomes.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Game Outcomes")
    plt.show()
```

---

## **4. Example Usage**  
```python
# Initialize environment
env = TicTacToeEnv()

# Log gameplay
print("Game Log:")
game_log = log_gameplay(env)
print(game_log)

# Visualize gameplay outcomes for multiple games
results = [1, -1, 0, 1, 0]  # Example results (X wins, O wins, draws)
visualize_outcomes(results)
```

---

## **Key Achievements**  
✔ **Custom Gymnasium Environment** – Implements Tic Tac Toe with RL capabilities.  
✔ **Game Logging** – Tracks actions, rewards, and game states.  
✔ **Player Switching** – Implements turn-based gameplay between X and O.  
✔ **Move Validation** – Prevents invalid or repeated moves.  
✔ **Board Visualization** – Displays board states in a structured format.  
✔ **Outcome Analysis** – Generates pie charts for win/draw distributions.  
✔ **Extensible for AI Training** – Can be integrated with RL agents like Q-learning.  

---

## **Conclusion**  
This implementation provides a fully functional Tic Tac Toe RL environment with built-in logging, visualizations, and game state management. It serves as a foundation for training reinforcement learning agents and analyzing optimal gameplay strategies. Future improvements can include training AI agents with Q-learning or Deep Q-Networks (DQN).

---

## **Future Work**  
🚀 Implement RL agent (Q-learning or DQN) for self-learning gameplay.  
🚀 Introduce configurable board sizes (e.g., 4x4 or 5x5).  
🚀 Enable multi-agent learning to optimize strategies.  

---
