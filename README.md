# **Overview of the Wolf-Sheep-Hunter Ecosystem Model**

The **Wolf-Sheep-Hunter Ecosystem Model** is a multi-agent simulation designed to model the complex interactions between **predators (wolves)**, **prey (sheep)**, and **humans (hunters)** within a bounded grid environment. The model is inspired by classical predator-prey systems, extended with an additional agent (hunter) to represent human intervention in controlling predator populations. The ecosystem also features **grass patches** that represent renewable resources for sheep.

## **Grid Environment**

- **Grid Size**: The environment is represented by a grid of dimensions (e.g., 50x50), where each cell can contain multiple agents, depending on their behavior and actions.
- **Torus Topology**: The environment can be represented as a torus (i.e., the grid wraps around, where the right edge connects with the left edge, and the top edge connects with the bottom). This creates a continuous, unbounded ecosystem for agents to navigate.

## **Agents in the Model**

The model includes four types of agents: **Sheep**, **Wolves**, **Hunters**, and **Grass Patches**. Each agent has its unique parameters, behaviors, and roles in the ecosystem.

### **1. Sheep**

**Role**: Sheep are **prey** in the ecosystem, feeding on **grass** to gain energy and attempting to avoid predators. Their goal is to **survive, reproduce**, and sustain their population.

- **Parameters**:
  - **Energy**: Initial energy level of the sheep (e.g., 10). Energy is required to move, reproduce, and survive.
  - **Perception Radius (`perception_r`)**: Radius within which sheep can detect wolves and hunters (e.g., 1 cell).
  - **Movement Types (`movement`)**:
    - **"step"**: Represents a normal move (1 cell per step).
    - **"flee"**: Represents a rapid movement to escape predators (2 cells per step).
  - **Reproduction Probability (`p_reproduction`)**: Probability with which a sheep reproduces if it has enough energy and is alone in a cell (e.g., 0.2).
  - **Energy Gain and Loss**:
    - **Energy Gain**: Sheep gain energy by eating **grass patches** (`eating_grass` gives, for example, 2 energy).
    - **Energy Loss**: Moving results in an energy loss (`step` consumes 1 energy, and `flee` consumes 5 energy).

- **Behavior**:
  - **Movement**: Sheep move randomly or flee when they sense a wolf in their perception radius.
  - **Feeding**: Sheep feed on grass when they encounter a fully grown **Grass Patch** in the cell they occupy. This helps them increase their energy level.
  - **Reproduction**: Sheep have a probability to reproduce if they have enough energy and occupy a cell alone.
  - **Death**: If energy falls to zero, sheep die, and they are removed from the environment.

### **2. Wolves**

**Role**: Wolves are the **predators** of the ecosystem, feeding on sheep to maintain their energy. Their goal is to **hunt, reproduce**, and sustain their population.

- **Parameters**:
  - **Energy**: Initial energy level of the wolf (e.g., 15). Wolves lose energy over time and need to hunt sheep to replenish their energy.
  - **Perception Radius (`perception_r`)**: Radius within which wolves can detect other agents, especially hunters (e.g., 3 cells).
  - **Movement Types (`movement`)**:
    - **"step"**: Normal movement (2 cells per step).
    - **"flee"**: Flee movement when wolves try to escape from hunters (e.g., 4 cells per step).
  - **Hunger Level**: Represents how urgently wolves need food, which influences their behavior.
  - **Reproduction Probability (`p_reproduction`)**: Probability with which a wolf reproduces if it has enough energy and is alone in a cell (e.g., 0.1).
  - **Energy Gain and Loss**:
    - **Energy Gain**: Wolves gain energy by eating sheep (`eating_pray` gives, for example, 6 energy).
    - **Energy Loss**: Wolves lose energy based on movement type (`step` loses 1 energy and `flee` loses 5 energy).

- **Behavior**:
  - **Movement**: Wolves move randomly to locate sheep or flee when a hunter is detected within their perception radius.
  - **Feeding**: Wolves feed on sheep when they share the same cell. They gain energy from eating a sheep, and the sheep is removed from the model.
  - **Reproduction**: Wolves reproduce with a specific probability when conditions allow, helping to maintain population levels.
  - **Death**: Wolves die and are removed from the environment if their energy reaches zero.

### **3. Hunters**

**Role**: Hunters are **human agents** responsible for controlling the wolf population. They can kill wolves either by **direct confrontation** or by placing **traps**.

- **Parameters**:
  - **Energy**: Initial energy level (e.g., 20). Hunters lose energy as they move and take action.
  - **Perception Radius (`perception_r`)**: Radius within which hunters can detect wolves to chase them (e.g., 3 cells).
  - **Movement (`movement`)**: Hunters move 1 cell per step, aiming to get closer to wolves if detected.
  - **Weapons (`weapons`)**:
    - **Bullets**: Used to kill wolves when sharing the same cell (`bullet` count starts at 10).
    - **Traps**: Placed in empty cells to trap and kill wolves later (`trap` count starts at 3).
  - **Energy Loss**: Moving, shooting, and setting traps have energy costs (`step` loses 1 energy and `put_trap` loses 2 energy).

- **Behavior**:
  - **Movement**: Hunters move towards wolves when detected, and otherwise move randomly.
  - **Hunting and Trapping**: Hunters kill wolves either by entering the same cell as a wolf or by setting traps. Traps are placed in empty cells, and when a wolf steps on the trap, it is killed.
  - **Death and Replacement**: Hunters lose energy as they move. When their energy reaches zero, they are temporarily removed from the model for two steps and then replaced to continue hunting.

### **4. Grass Patches**

**Role**: Grass patches represent renewable resources that **sheep** need to survive.

- **Parameters**:
  - **Fully Grown**: Indicates if the grass patch is available for consumption.
  - **Regrowth Time**: Grass takes time to regrow after being eaten by sheep.

- **Behavior**:
  - **Growth and Regrowth**: When sheep eat the grass, it becomes unavailable, and a countdown for regrowth begins. Once the regrowth time is complete, the grass becomes fully grown again.


## **Metrics and Data Collection**

To monitor and analyze the ecosystem's dynamics, the model collects the following metrics:

- **Population Counts**: The number of **Sheep**, **Wolves**, and **Hunters** at each time step.
- **Kill Metrics**:
  - **Wolf Kills**: Number of sheep killed by wolves.
  - **Hunter Kills**: Number of wolves killed by hunters.
- **Deaths by Energy**: The number of **Sheep**, **Wolves**, or **Hunters** that die due to running out of energy.
- **Energy Metrics**: Distribution of energy levels across agent types can also be visualized to show health and sustainability.


