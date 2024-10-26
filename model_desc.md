# Detailed Ecosystem Model Overview

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


## Policy Dynamics and Their Effects on Hunters

The simulation incorporates a sophisticated policy-making mechanism driven by public opinion, which in turn is influenced by the ongoing interactions and conflicts within the ecosystem. These policies directly affect the capabilities and behaviors of hunters in the simulation, representing real-world wildlife management decisions. Below is a detailed explanation of each policy and its impact on hunter operations.

### Policy Types

1. **Increase Hunter Restrictions**
   - **Description**: This policy is enacted when public concern about wolf populations is low. The aim is to limit hunter activities to ensure that wolf populations are not unduly diminished, promoting ecological balance.
   - **Effects on Hunters**:
     - **Reduced Movement**: Hunters have their movement radius decreased, limiting their ability to cover large areas quickly. This simulates increased regulations or restricted access to certain areas within the simulation grid.


2. **Maintain Normal Operations**
   - **Description**: This default policy is applied when there is a moderate level of public concern. It reflects a balanced approach to wildlife management, allowing hunters to operate without new restrictions but without additional support or capabilities.
   - **Effects on Hunters**:
     - **Standard Movement and Activities**: Hunters continue with their predefined parameters for movement. This represents a steady state of wildlife management where existing rules and capabilities are deemed adequate.

3. **Decrease Hunter Restrictions**
   - **Description**: Activated when there is high public concern regarding wolf activity, possibly due to increased conflicts or perceived threats from wolf populations. This policy allows for more aggressive management tactics to control these populations.
   - **Effects on Hunters**:
     - **Increased Movement**: Hunters are allowed to move more freely and cover more ground, simulating an easing of restrictions that might come with heightened concerns about wildlife.
    
### Implementation and Feedback Loop

Each policy is implemented based on the current state of public opinion, which is itself influenced by the ongoing dynamics of the ecosystem, particularly the conflict score. As wolves interact with human habitats and livestock, they can either increase or decrease this score, which in turn influences public opinion. Changes in public opinion trigger reassessments of policies, creating a dynamic feedback loop that continuously adapts to the unfolding ecological interactions.

- **Monitoring and Adjustment**: The effects of each policy are monitored through the simulation's data collection mechanisms. Adjustments to policies can be made in response to their observed impacts on the ecosystem, ensuring that the model remains responsive to the needs of both the human and wildlife populations it simulates.

This dynamic policy framework adds a layer of depth to the simulation, allowing users to explore the consequences of different wildlife management strategies and their effectiveness in real-time. By adjusting the model's parameters and observing the resulting changes in hunter behavior and overall ecosystem health, stakeholders can gain insights into the complex interplay of ecological factors and human activities.

### Simulation Mechanics

#### Conflict Score Dynamics
- **Score Increase**: Occurs when wolves interact negatively with livestock or human habitats, which can be adjusted based on the proximity and density of human populations.
- **Score Decrease**: Implemented to reduce the conflict score when wolves remain in low-conflict areas, promoting strategies that minimize human-wildlife conflicts.

#### Policy Influence
- **Policy Adjustments**: The policy maker adjusts hunting restrictions based on public opinion, which is influenced by the current conflict score. These policies can increase or decrease hunter mobility and efficiency.

#### Population Management
- **Wolf Pack Dynamics**: Rules governing when wolf packs split or are culled based on their sizes and interactions with hunters, with parameters defining the thresholds for these actions.

  ### Human Density Dynamics

The simulation integrates a sophisticated human density model to represent the spatial distribution of human populations within the ecosystem. This feature is crucial for mimicking real-world scenarios where wildlife interactions with human populations vary significantly based on the density of human habitation. Here's how human density is modeled and its impact on the ecosystem:

### Representation of Human Density

- **Human Habitat Agents**: Each grid cell may contain a `HumanHabitatAgent` that represents the human population in that area. The density of these agents across the grid mirrors variations in human population density, ranging from urban centers to rural areas.
- **Density Map Initialization**: Human density is not randomly distributed but is instead initialized based on a set of predefined centers that represent towns or cities. The density decreases as the distance from these centers increases, simulating the typical drop-off in population density seen in real-world urban-rural gradients.


## Visualization and Data Collection
The simulation uses various modules to visualize and collect data on the ecosystem:

- **CanvasGrid**: Shows the position and state of each agent with distinct colors and shapes.
- **DataCollector**: Captures and records detailed simulation data at each step, suitable for analysis and review.
- **TextElement and ChartModule**: Provides updates on key metrics such as wolf pack numbers, conflict scores, and policy states in real-time.

## Conclusion

By adjusting the parameters described, users can simulate a wide range of ecological scenarios to study the impacts of different management strategies and environmental conditions. This flexibility makes the Ecosystem Simulation a powerful tool for educational purposes, ecological research, and policy-making.

