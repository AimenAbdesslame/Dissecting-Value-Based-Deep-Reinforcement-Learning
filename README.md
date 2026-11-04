# Dissecting Value-Based Deep Reinforcement Learning: Target Networks, Experience Replay, and Dueling Double DQN in Classical, Visual, and Sparse-Reward Tasks

**Course:** Reinforcement Learning (3rd Year IASD, ESI-SBA)  
**Academic Supervision:** Pr. Abdelhamid Malki  
**Author:** Aimen Abdesselam  
**Evaluation Protocol:** Seeds {0, 1, 2}, Discount Factor $\gamma = 0.99$, Greedy Evaluation over 20 Unseen Seeds ($100\,000 + i$, $\varepsilon = 0$). Results reported as $\text{mean } [\min, \max]$.

---

## Abstract

This repository provides an empirical investigation into value-based Deep Reinforcement Learning (DRL), focusing on stability mechanics, algorithmic ablations, and architectural extensions of the Deep Q-Network (DQN) framework. We evaluate the core components of DQN—specifically Experience Replay and Target Network synchronization protocols—across continuous control and visual observation spaces. Furthermore, we benchmark decoupled Double DQN (DDQN) and Dueling DQN architectures on classical control benchmarks (CartPole-v1, LunarLander-v3), high-dimensional visual environments (ALE/Pong-v5 under an Atari preprocessing pipeline), and a custom non-stationary gridworld navigation task (TreasureGrid-v0) with moving obstacles and sparse reward formulations.

---

## 1. Environments and System Configurations

Experiments span four distinct environments covering low-dimensional physical state spaces, pixel-based frame stacks, and tabular POMDP-like navigation:

| Environment | Observation Space | Action Space | Network Topology | Optimizer / Batch | Synchronization ($C$ / $\tau$) | Exploration Schedule | Termination Budget | Solved Criterion |
|---|---|:---:|---|---|---|---|---|:---:|
| **CartPole-v1** | $\mathbb{R}^4$ | 2 | MLP: $4 \to 64 \to 64 \to 2$ | Adam ($5 \times 10^{-4}$), $B=64$ | $C = 100$ steps / $\tau = 0.01$ | $\varepsilon: 1.0 \to 0.05$ ($10\text{k}$ steps) | 300 episodes | $\ge 475$ |
| **LunarLander-v3** | $\mathbb{R}^8$ | 4 | MLP: $8 \to 128 \to 128 \to 4$ | Adam ($5 \times 10^{-4}$), $B=64$ | $C = 250$ steps / $\tau = 0.01$ | $\varepsilon: 1.0 \to 0.05$ ($25\text{k}$ steps) | 800 episodes | $\ge 200$ |
| **ALE/Pong-v5** | $4 \times 84 \times 84$ (`uint8`) | 6 | Nature CNN $\to$ FC $512 \to 6$ | Adam ($1 \times 10^{-4}$), $B=32$ | $C = 1000$ steps / $\tau = 0.01$ | $\varepsilon: 1.0 \to 0.01$ ($200\text{k}$ steps) | $200\text{k}$ steps* | $\ge +18.0$ |
| **TreasureGrid-v0** | $\mathbb{R}^8$ | 4 | MLP: $8 \to 64 \to 64 \to 4$ | Adam ($5 \times 10^{-4}$), $B=64$ | $C = 100$ steps / $\tau = 0.01$ | $\varepsilon: 1.0 \to 0.05$ ($10\text{k}$ steps) | 500 episodes | Success Rate |

*\*Note on Pong budget:* To accommodate constrained compute budgets on cloud accelerators (e.g., Kaggle T4 GPUs), Pong training steps were standardized to $200\,000$ environment steps ($800\,000$ frames with frame skip 4) across all variants.

---

## 2. Part 1: DQN Component Ablation Study

To isolate the individual contribution of experience replay and target networks to training stability and value convergence, four configurations were evaluated under strictly identical hyperparameter profiles:

1. **DQN Hard (Reference):** Periodic parameter assignment $\theta^- \leftarrow \theta$ every $C$ gradient steps.
2. **DQN Soft Update:** Polyak averaging at every step: $\theta^- \leftarrow \tau \theta + (1 - \tau)\theta^-$ with $\tau = 0.01$.
3. **No Replay Buffer:** Optimization performed strictly on the last $B$ consecutive transitions sampled directly from online rollouts, retaining serial correlation.
4. **No Target Network:** Target calculation uses current online weights: $Y_t = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a; \theta_t)$.

### 2.1 Empirical Results

```
========================================================================================================
Environment      Metric         DQN Hard (Ref)          DQN Soft (tau=0.01)    No Replay               No Target Net
========================================================================================================
CartPole-v1      Train Mean     266.8 [248.0, 290.0]    371.4 [336.0, 405.0]   342.1 [293.0, 412.0]    286.8 [258.0, 307.0]
                 Greedy Eval    416.8 [250.0, 500.0]    500.0 [500.0, 500.0]   330.4 [231.0, 500.0]    282.6 [119.0, 444.0]
--------------------------------------------------------------------------------------------------------
LunarLander-v3   Train Mean     227.6 [213.0, 250.0]    219.2 [209.0, 232.0]   -73.2 [-93.0, -38.0]    236.1 [231.0, 240.0]
                 Greedy Eval    245.5 [232.0, 261.0]     63.5 [-196.0, 245.0]  -91.9 [-150.0, -54.0]   216.1 [174.0, 242.0]
--------------------------------------------------------------------------------------------------------
ALE/Pong-v5      Train Mean     -17.9 [-20.0, -15.0]    -12.0 [-14.0, -10.0]   -20.6 [-21.0, -20.0]    -19.1 [-21.0, -17.0]
                 Greedy Eval    -15.5 [-19.0, -12.0]    -12.1 [-16.0, -9.0]    -21.0 [-21.0, -21.0]    -18.6 [-21.0, -14.0]
========================================================================================================
```

### 2.2 Analytical Findings

* **Criticality of Experience Replay:** In complex dynamics (LunarLander) and pixel domains (Pong), removing the replay buffer induces catastrophic policy collapse ($-91.9$ on LunarLander and $-21.0$ on Pong). Non-stationary data distributions and temporally correlated gradients prevent convergence.
* **Target Network Stabilization:** In CartPole-v1, removing the target network triggers severe Q-value divergence. Predicted values inflate to $\sim 310$, exceeding the theoretical maximum infinite-horizon discounted return:
  $$\sum_{t=0}^{\infty} \gamma^t R_{\max} = \frac{1}{1 - \gamma} = \frac{1}{1 - 0.99} = 100$$
  Both Hard and Soft target synchronization restrain Q-values to the theoretical boundary of $100$.
* **Polyak Averaging vs. Periodic Assignment:** Soft updates ($\tau = 0.01$) achieve robust performance on CartPole-v1 (solved at $500.0$ across 3/3 seeds) and accelerate sample efficiency on high-dimensional visual inputs (leading Pong at $200\text{k}$ steps with $-12.1$). However, soft updates demonstrate variance on LunarLander (seed crash to $-196.0$).

---

## 3. Part 2: Double DQN and Dueling DQN Architectures

### 3.1 Mathematical Formulations

Standard DQN uses identical parameters to select and evaluate actions, maximizing over estimated action values:
$$Y_t^{\text{DQN}} = R_{t+1} + \gamma (1 - d_t) \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

#### Double DQN (DDQN)
Decouples action selection from action evaluation to counteract positive maximization bias:
$$a^* = \arg\max_{a'} Q(S_{t+1}, a'; \theta)$$
$$Y_t^{\text{Double}} = R_{t+1} + \gamma (1 - d_t) Q(S_{t+1}, a^*; \theta^-)$$

#### Dueling DQN
Separates state value $V(s)$ from state-dependent action advantages $A(s, a)$, sharing a common feature trunk and combining through a mean-centered identifiability constraint:
$$Q(S, a; \theta, \alpha, \beta) = V(S; \theta, \beta) + \left( A(S, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(S, a'; \theta, \alpha) \right)$$

### 3.2 Empirical Results

```
========================================================================================================
Environment      Metric         DQN                     Double DQN              Dueling DQN             Dueling Double DQN
========================================================================================================
CartPole-v1      Train Mean     266.8 [248.0, 290.0]    280.2 [242.0, 315.0]    340.0 [270.0, 381.0]    301.9 [264.0, 366.0]
                 Greedy Eval    416.8 [250.0, 500.0]    471.7 [415.0, 500.0]    402.7 [299.0, 500.0]    375.2 [268.0, 500.0]
--------------------------------------------------------------------------------------------------------
LunarLander-v3   Train Mean     227.6 [213.0, 250.0]    238.1 [225.0, 249.0]    225.2 [210.0, 237.0]    222.8 [212.0, 232.0]
                 Greedy Eval    245.5 [232.0, 261.0]    234.1 [226.0, 249.0]    234.7 [209.0, 257.0]    211.1 [194.0, 243.0]
--------------------------------------------------------------------------------------------------------
ALE/Pong-v5      Train Mean     -17.6 [-20.0, -13.1]    -18.2 [-20.4, -13.8]    -11.4 [-12.6, -10.6]    -17.9 [-20.4, -13.3]
                 Greedy Eval    -18.0 [-21.0, -12.0]    -18.1 [-21.0, -12.2]     -4.0 [-9.2, -1.1]      -17.8 [-21.0, -11.3]
========================================================================================================
```

### 3.3 Overestimation and Architectural Impact

* **Mitigation of Maximization Bias:** In CartPole-v1, baseline DQN overestimates Q-values to $125\text{--}130$. Dueling architectures enforce strict stabilization around $95\text{--}100$, tracking the theoretical ceiling without divergence.
* **Breakthrough on Visual Observations:** In ALE/Pong-v5, Dueling DQN demonstrates substantial sample efficiency advantages, attaining a greedy score of $-4.0$ within $200\text{k}$ steps (reaching returns up to $-1.1$). Standard DQN, Double DQN, and Dueling Double DQN remain constrained near $-18.0$ under identical budgets. Advantage stream separation allows rapid general state value updates without requiring exhaustive exploration across every action per frame.

---

## 4. Part 3: Custom Environment — TreasureGrid-v0

### 4.1 Environment Specification
A fully custom Gymnasium-compliant grid environment designed to examine reward sparsity and non-stationary dynamic hazards:

* **Grid Dimensions:** $6 \times 6$ toroidal-free grid with interior structural walls.
* **Entities:** Static starting position $(0, 0)$, dynamic hazards (2 moving traps with stochastic random-walk transitions), and target treasure $(5, 5)$.
* **State Representation:** Continuous normalized vector $\mathbb{R}^8$:
  $$S = [x_{\text{agent}}, y_{\text{agent}}, x_{\text{treasure}}, y_{\text{treasure}}, x_{\text{trap1}}, y_{\text{trap1}}, x_{\text{trap2}}, y_{\text{trap2}}] \in [0, 1]^8$$
* **Action Space:** Discrete (4 directions: Up, Down, Left, Right).
* **Episode Horizon:** 144 steps timeout.
* **Compliance:** Passes `gymnasium.utils.env_checker.check_env` across all variants.

### 4.2 Reward Variants

1. **Version 1 (Sparse Reward):** Step penalty $-0.05$, illegal wall collision penalty $-0.2$, terminal reward $+10.0$ (Treasure) and terminal penalty $-10.0$ (Trap).
2. **Version 2 (Dense Reward):** Potential-shaped distance metric:
   $$R_t = -0.05 - 0.01 \times \|(x_{\text{agent}}, y_{\text{agent}}) - (x_{\text{treasure}}, y_{\text{treasure}})\|_1$$
   Terminal conditions retain $\pm 10.0$.
3. **Version 3 (Randomized Goal):** Reward profile of V1 with stochastic goal repositioning sampled uniformly across free cells at each reset.

### 4.3 Benchmark Comparison Against Baselines

Baselines:
* **Random Agent:** Uniform random action selection over episode budgets.
* **BFS Oracle:** Breadth-First Search computing shortest path from agent to treasure ignoring moving traps (theoretical deterministic optimum).

```
========================================================================================================
Configuration       Metric                     BFS Oracle    Uniform Random    DQN          Double DQN     Dueling DQN
========================================================================================================
V1: Sparse          Greedy Return              3.40          -3.04             -13.24       -0.99          -12.60
                    Success Rate (%)           100.0%        40.0%             0.0%         51.7%          0.0%
                    Mean Episode Steps         3.4           13.4              144.0        69.6           144.0
--------------------------------------------------------------------------------------------------------
V2: Dense           Greedy Return              3.40          -6.86             -11.59       -6.95          -11.81
                    Success Rate (%)           100.0%        25.0%             0.0%         18.3%          0.0%
                    Mean Episode Steps         3.4           21.1              144.0        118.1          144.0
--------------------------------------------------------------------------------------------------------
V3: Random Goal     Greedy Return              4.30          -5.62             -8.93        -5.58          -12.25
                    Success Rate (%)           100.0%        30.0%             11.7%        23.3%          8.3%
                    Mean Episode Steps         4.3           26.2              127.6        105.0          132.6
========================================================================================================
```

### 4.4 Failure Modes and Mechanistic Analysis

* **Failure of DQN and Dueling DQN on Sparse Grids:** Both DQN and Dueling DQN register $0.0\%$ success rates on V1 and V2, becoming trapped in deterministic limit cycles (oscillating against walls until timing out at step 144). Stochastic transitions induced by moving traps cause severe overestimation of non-terminal loop paths.
* **Superiority of Double DQN:** Double DQN breaks self-reinforcing overestimation loops, achieving a **51.7% success rate** on V1 Sparse, outperforming both naive exploration ($40.0\%$) and alternative architectures.
* **The Dense Reward Trap:** Potential-based distance shaping (V2) degrades Double DQN performance ($18.3\%$ vs $51.7\%$). The continuous dense penalty gradient encourages passive evasion of moving hazards, causing agents to delay goal progression.

---

## 5. Repository Structure

```
.
├── .gitignore                                 # Excludes build, bytecode, and cache artifacts
├── README.md                                  # Academic summary and benchmark documentation
├── DQN_Homework_Aimen_Abdesselam.pptx         # Academic presentation deck
├── treasure_grid.py                           # Gymnasium custom environment implementation
│
├── ex1-ablation-cartpole-kaggle.ipynb         # EX1: Ablation studies on CartPole-v1
├── ex1-ablation-lunarlander.ipynb             # EX1: Ablation studies on LunarLander-v3
├── ex1-ablation-pong.ipynb                    # EX1: Ablation studies on ALE/Pong-v5
├── ex2-ddqn-dueling-cartpole-kaggle.ipynb     # EX2: Double / Dueling DQN on CartPole-v1
├── ex2-ddqn-dueling-lunarlander.ipynb         # EX2: Double / Dueling DQN on LunarLander-v3
├── ex2-ddqn-dueling-pong.ipynb                # EX2: Double / Dueling DQN on ALE/Pong-v5
├── ex3-customenv.ipynb                        # EX3: TreasureGrid benchmark and baselines
│
└── outputs/                                   # Checkpoints (.pth) and learning curves (.png)
    ├── EX1_Ablation_CartPole-kaggle/
    ├── EX1_Ablation_LunarLander/
    ├── EX1_Ablation_Pong/
    ├── EX2_DDQN_Dueling_CartPole-Kaggle/
    ├── EX2_DDQN_Dueling_LunarLander/
    ├── EX2_DDQN_Dueling_Pong/
    └── EX3_CustomEnv/
```

---

## 6. Reproduction and Execution

### 6.1 Local Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/deep-q-learning-ablation-benchmarks.git
cd deep-q-learning-ablation-benchmarks

pip install torch torchvision numpy matplotlib gymnasium[box2d] gymnasium[atari] ale-py opencv-python autorom[accept-rom-license]
```

Validate the custom environment:

```python
import gymnasium as gym
import treasure_grid

for v in [1, 2, 3]:
    env = gym.make("TreasureGrid-v0", version=v)
    gym.utils.env_checker.check_env(env.unwrapped)
    print(f"TreasureGrid-v{v} passed Gymnasium check_env.")
```

### 6.2 Execution Guidelines

* **Low-Dimensional Tasks (CartPole, LunarLander, TreasureGrid):** Execute directly on CPU or cloud instances. Full 3-seed runs complete within 15–30 minutes per notebook.
* **Atari Environments (ALE/Pong-v5):** GPU acceleration (NVIDIA T4 or higher) is required. The $200\,000$ step budget executes in approximately 3 to 4 hours per notebook.

---

## 7. Conclusions

1. **Experience Replay is Indispensable:** Without off-policy replay decorrelation, gradient updates diverge on complex physical control and visual inputs.
2. **Double DQN Provides Consistent Regularization:** Decoupled action selection reliably eliminates value overestimation and prevents terminal looping in non-stationary stochastic environments.
3. **Dueling Architectures Accelerate Visual Learning:** Decoupling state value $V(s)$ from action advantages $A(s, a)$ enables superior representation learning from pixel streams where actions do not influence every background transition.
