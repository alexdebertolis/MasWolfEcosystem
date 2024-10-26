# Detailed Ecosystem Simulation Overview

The Ecosystem Simulation, developed using the Mesa framework, models interactions among wolves, hunters, livestock, and human habitats in a dynamic environment influenced by real-world geographic data. This detailed overview focuses on the adjustable parameters and dynamics that allow researchers and policymakers to explore different ecological scenarios effectively.

## Simulation Environment

### Grid Initialization
The simulation's environment is based on a **ShapefileGrid** class which loads valid geographic cells from a shapefile. This integration allows the simulation to reflect real-world geography, affecting the movement and interactions of agents:

- **Valid Cells**: Agents can only be placed and move within these defined cells, which are extracted from the geographic centroids in the shapefile.
- **Geographic Boundaries**: The simulation's spatial boundaries are determined by the extents of the loaded shapefile, ensuring that all activities occur within realistic geographical limits.

### Agents
The simulation includes several types of agents, each with specific behaviors and characteristics:

- **Wolf Packs**: Represent predator dynamics, capable of moving towards prey, reproducing, and engaging in conflicts with humans and livestock. 
- **Hunters**: Simulate human predator control activities, influenced by ecological policies and their own survival strategies.
- **Livestock and Human Habitats**: Static agents that contribute to the ecological balance by creating conflict or coexistence scenarios.
- **Public Opinion and Policy Maker**: Non-physical agents that drive policy changes based on ecological outcomes, affecting hunter restrictions and overall wildlife management.

## Key Parameters and Dynamics

### Initialization Parameters
These parameters are critical as they directly influence the initial state and progression of the simulation:

- **`livestock_ratio`**: Sets the percentage of grid cells that contain livestock at the start of the simulation, affecting initial conflict zones.
- **`N_wolfpacks` and `N_hunters`**: Determine the initial number of wolf packs and hunters, crucial for establishing the predator-prey dynamics.
- **`num_density_centers`**: Dictates the number of high-density human population centers, influencing the spatial distribution of human agents and their interactions with wildlife.

### Agent-Specific Parameters
Each agent type has parameters that dictate its behavior within the ecosystem:

- **Perception Radius**: Defines the range within which wolves and hunters can detect other agents or elements of interest, crucial for their movement decisions.
- **Reproduction Probability**: The likelihood of wolves reproducing each simulation step, impacting the growth rate of wolf populations.
- **Energy and Respawn Mechanics for Hunters**: Influence how long hunters can operate before needing recovery, simulating real-world fatigue and resource management.

### Simulation Mechanics

#### Conflict Score Dynamics
- **Score Increase**: Occurs when wolves interact negatively with livestock or human habitats, which can be adjusted based on the proximity and density of human populations.
- **Score Decrease**: Implemented to reduce the conflict score when wolves remain in low-conflict areas, promoting strategies that minimize human-wildlife conflicts.

#### Policy Influence
- **Policy Adjustments**: The policy maker adjusts hunting restrictions based on public opinion, which is influenced by the current conflict score. These policies can increase or decrease hunter mobility and efficiency.

#### Population Management
- **Wolf Pack Dynamics**: Rules governing when wolf packs split or are culled based on their sizes and interactions with hunters, with parameters defining the thresholds for these actions.

## Visualization and Data Collection
The simulation uses various modules to visualize and collect data on the ecosystem:

- **CanvasGrid**: Shows the position and state of each agent with distinct colors and shapes.
- **DataCollector**: Captures and records detailed simulation data at each step, suitable for analysis and review.
- **TextElement and ChartModule**: Provides updates on key metrics such as wolf pack numbers, conflict scores, and policy states in real-time.

## Conclusion

By adjusting the parameters described, users can simulate a wide range of ecological scenarios to study the impacts of different management strategies and environmental conditions. This flexibility makes the Ecosystem Simulation a powerful tool for educational purposes, ecological research, and policy-making.

