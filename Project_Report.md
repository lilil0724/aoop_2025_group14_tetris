# Tetris Battle: A Python-based Multiplayer Tetris Game with Weighted Heuristic AI

**Abstract**—This paper presents the design and implementation of "Tetris Battle," a modern clone of the classic Tetris game developed using Python and Pygame. The project features a robust object-oriented architecture supporting multiple game modes, including single-player, local multiplayer, LAN-based network multiplayer, and Player vs. AI. A key contribution of this work is the implementation of a highly efficient Artificial Intelligence (AI) agent based on a Weighted Heuristic algorithm (Dellacherie’s Algorithm), which outperforms traditional Deep Reinforcement Learning (DRL) approaches in terms of computational efficiency and explainability for this specific domain. The system utilizes TCP/IP sockets for low-latency network synchronization and Numpy for optimized matrix operations.

## I. Introduction

Tetris, since its inception, has been a benchmark for puzzle games and a popular subject for Artificial Intelligence research. While many implementations exist, few combine modern competitive mechanics (such as "Tetris Battle" rules) with a lightweight, high-performance AI agent in a unified Python environment.

The objective of this project is to develop a feature-rich Tetris game that replicates the competitive experience of modern Tetris titles. The system is designed to support:
1.  **Versatile Gameplay**: Solo practice, local 1v1, and networked LAN battles.
2.  **Competitive Mechanics**: Implementation of the "7-Bag" randomization system, Ghost Piece prediction, Hold functionality, and garbage line attack mechanics.
3.  **Intelligent AI**: A computer opponent capable of playing at a high level without the heavy computational overhead of neural networks.

This paper details the system architecture, the rationale behind the AI algorithm selection, and the implementation of network synchronization.

## II. System Architecture

The project is built using **Python 3.11**, leveraging **Pygame** for the graphical user interface and event handling, and **Numpy** for efficient grid manipulations.

### A. Object-Oriented Design
The system follows a strict Object-Oriented Programming (OOP) paradigm to ensure modularity and maintainability. The core class structure includes:

*   **`GameEngine`**: The central controller that manages the game loop, state transitions (Menu, Game, Game Over), and coordinates updates between players.
*   **`PlayerContext`**: Encapsulates all runtime data for a single player (board state, current piece, score, AI agent). This allows for easy scalability from single-player to multi-player modes by simply instantiating multiple contexts.
*   **`Handler`**: Implements the core game logic, separating data (`Shot`, `Piece`) from behavior. It handles collision detection, rotation rules (SRS - Super Rotation System), and line clearing.
*   **`Shot` & `Piece`**: Data classes representing the static board matrix and the active falling tetromino, respectively.
*   **`NetworkManager`**: Manages TCP/IP connections, packet serialization, and state synchronization.

### B. Networking
The multiplayer component uses Python's `socket` library to establish a Client-Server architecture over LAN.
*   **Protocol**: TCP (Transmission Control Protocol) is used with `TCP_NODELAY` to minimize latency.
*   **Synchronization**: Game states (board configuration, active piece, garbage lines) are serialized using `pickle` and broadcasted to connected clients.
*   **Concurrency**: Threading is employed to handle network I/O without blocking the main rendering loop, ensuring a smooth 60 FPS experience.

## III. AI Implementation

A significant component of this project is the AI agent. While Deep Reinforcement Learning (DRL) methods like Deep Q-Networks (DQN) are popular, this project adopts a **Weighted Heuristic Search** approach, specifically a variation of **Dellacherie’s Algorithm**.

### A. Algorithm Selection
We compared Deep RL against Weighted Heuristics and chose the latter for the following reasons:
1.  **Explainability**: Heuristic weights directly correspond to game concepts (e.g., "minimize holes"), whereas Neural Networks act as "black boxes."
2.  **Performance**: The heuristic approach requires only simple matrix operations, allowing it to run in real-time (under 1ms per decision) on standard CPUs, unlike DRL which often requires GPU acceleration for inference.
3.  **Stability**: DRL agents in Tetris often suffer from "catastrophic forgetting" or learn sub-optimal strategies (e.g., suicide to avoid penalties) if the reward function is not perfectly tuned.

### B. Feature Engineering
The AI evaluates every possible move (position and rotation) for the current piece and selects the one that maximizes a utility function based on 8 features:
1.  **Landing Height**: The height at which the piece is placed (minimize).
2.  **Max Height**: The highest occupied cell on the board (minimize).
3.  **Holes**: Empty cells covered by filled cells (heavily penalized, weight $\approx -7.9$).
4.  **Row Transitions**: The number of filled-to-empty transitions in rows (minimize for smoothness).
5.  **Column Transitions**: The number of filled-to-empty transitions in columns.
6.  **Well Sums**: A measure of "wells" (deep vertical channels) that are necessary for Tetris line clears.
7.  **Deep Wells**: Wells deeper than 2 blocks (penalized to prevent unrecoverable situations).
8.  **Cumulative Wells**: A weighted sum of all well depths.

The AI calculates the score for all legal moves and executes the one with the highest score.

## IV. Game Mechanics

To ensure a competitive experience, the game implements standard modern Tetris guidelines:
*   **7-Bag Randomizer**: Ensures a balanced distribution of piece types, preventing long droughts of specific shapes (like the 'I' piece).
*   **Ghost Piece**: Displays a visual guide for where the piece will land, improving player accuracy.
*   **Garbage System**: In multiplayer modes, clearing lines sends "garbage lines" to the opponent's board. A "Back-to-Back" (B2B) bonus is implemented to reward consecutive difficult clears (like Tetris or T-Spins).

## V. Conclusion

This project successfully demonstrates a full-featured Tetris Battle game implemented in Python. By utilizing a modular OOP architecture, we achieved a flexible system that supports various game modes. The integration of a Weighted Heuristic AI provides a challenging opponent that is both computationally efficient and strategically competent. Future work could involve implementing a matchmaking server for wide-area network (WAN) play and using Genetic Algorithms to further fine-tune the AI's weights.

## References

[1] C. Fahey, "Tetris AI – The Near Perfect Player," [Online]. Available: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/.
[2] P. Dellacherie, "Pierre Dellacherie’s Algorithm," [Online].
[3] Python Software Foundation, "Python 3.11 Documentation," 2023.
[4] Pygame Community, "Pygame Documentation," [Online]. Available: https://www.pygame.org/docs/.
