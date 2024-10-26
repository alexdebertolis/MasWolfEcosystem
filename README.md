# Detailed Overview of the Ecosystem Simulation Model

The Ecosystem Simulation Model is a complex agent-based model designed to simulate interactions between different agents: wolves, hunters, livestock, human habitats, public opinion, and policy makers within a geographically accurate grid environment. This model explores the dynamics of predator-prey relationships, human-wildlife conflicts, and the impact of policy decisions on these interactions.

## Simulation Environment

### Grid Initialization

- **ShapefileGrid**: This class initializes the grid based on a shapefile, loading only the valid cells where agents can exist, which are determined from the geographic centroids provided in the shapefile. This ensures that the simulation reflects realistic geographical constraints.
- **Grid Size and Boundaries**: The grid dimensions are derived from the shapefile's bounds, creating a spatially accurate representation of the real-world area being simulated.

### Agents

The model includes a variety of agents each with specific roles and behaviors:

#### Wolves (WolfPack Agent)
- **Role**: Predators that interact with livestock and human habitats, affecting the conflict score within the simulation.
- **Behaviors**: Wolves can move towards prey, reproduce based on a probability factor, and split into new packs if their size exceeds a threshold.
- **Parameters**:
  - **Perception Radius**: Determines how far wolves can detect prey or threats.
  - **Reproduction Probability**: Influences how often wolves can reproduce, adding new wolves to the pack.
  - **Pack Size Management**: Wolves manage their pack size through reproduction and can split into new packs if the size becomes too large.

#### Hunters
- **Role**: Represent human intervention in wildlife management aiming to control wolf populations.
- **Behaviors**: Hunters can move around the grid, kill wolves, set traps, and respond to policy changes influenced by public opinion.
- **Parameters**:
  - **Energy and Respawn Mechanics**: Control how long hunters can operate before needing to respawn.
  - **Policy Response**: Hunters adjust their behavior based on the current policy, which can restrict or enhance their hunting capabilities.

#### Livestock and Human Habitats
- **Static Agents**: Influence the conflict score when wolves interact with them.

#### Public Opinion and Policy Maker
- **Dynamic Influence**: These agents react to the unfolding events within the simulation and adjust policies accordingly which in turn affect hunter behaviors.

## Key Simulation Mechanics

### Conflict Score Dynamics
- **Adjustment Mechanisms**: The conflict score increases with negative interactions between wolves and humans or livestock, and decreases when wolves stay in low-conflict zones.
- **Influence on Policies**: The conflict score directly influences public opinion, which in turn affects policy decisions.

### Policy Dynamics
- **Policy Adjustments**: Based on public opinion, policies can become stricter or more lenient, directly influencing hunter behaviors and indirectly affecting wolf population dynamics. A log_communication method print a txt file with the xml communication between public - policy maker - hunter.

### Agent Interactions
- **Predator-Prey Dynamics**: Wolves hunt livestock, affecting the conflict score and potentially triggering policy changes through shifts in public opinion.
- **Hunter-Wolf Interactions**: Hunters seek out and reduce wolf populations, influenced by the prevailing policy settings.

## Simulation Parameters

### Adjustable Parameters for Experimentation
These parameters can be adjusted to explore different scenarios and outcomes within the simulation:

- **Livestock Ratio**: Changes the initial distribution of livestock across the grid.
- **Initial Number of Agents**: The starting number of wolves, hunters, and the distribution of human habitats.
- **Human Density Centers**: Adjusts the number of high-density human population centers, affecting human-wolf interactions.
- **Perception Radius and Movement Parameters for Agents**: Can be tuned to simulate different behaviors and interaction outcomes.

### Visualization and Data Collection

- **CanvasGrid Visualization**: Displays the position and status of each agent, updating in real-time as the simulation progresses.
- **DataCollector**: Captures detailed data on wolf and hunter populations, conflict scores, and policy changes at each simulation step for further analysis.
- **Real-Time Text Elements**: Display current data on public opinion, policies, and wolf pack counts, enhancing the understanding of the simulation's state and dynamics.

## Conclusion

This Ecosystem Simulation Model offers a robust platform for exploring the complex dynamics of wildlife management and human-wildlife coexistence. By adjusting various parameters, users can simulate and visualize different management strategies and their potential impacts on the ecosystem, providing valuable insights for researchers, educators, and policymakers.
