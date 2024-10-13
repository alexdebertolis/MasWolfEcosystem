import random
from IPython import display
from ipywidgets import widgets
import mesa
from mesa.space import MultiGrid
import seaborn as sns
import numpy as np
import pandas as pd
#from mesa.visualization.modules import CanvasGrid, ChartModule
#from mesa.visualization.ModularVisualization import ModularServer
import matplotlib.pyplot as plt
from mesa.experimental import JupyterViz
from mesa.experimental import make_text
import solara

def manhattan_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def agent_portrayal(agent):
    size = 10
    color = "gray"  # Default color for unknown agent type
    shape = "rect"

    if isinstance(agent, Sheep):
        size = 20
        color = "green" if agent.energy > 5 else "tab:olive"  # Represent sheep differently depending on energy
    elif isinstance(agent, Wolf):
        size = 30
        color = "red" if agent.energy > 5 else "tab:orange"  # Represent wolves differently based on energy level
    elif isinstance(agent, Hunter):
        size = 40
        color = "blue"  # Hunters have a fixed color representation
    """elif isinstance(agent, GrassPatch):
        if agent.fully_grown:
            size=20
            color ="green"
            shape="rect"

        else:
            size=20
            color ="brown"
            shape="rect"""""

    return {"size": size, "color": color, "shape" : shape}
"""def agent_portrayal(agent):
   
    # Default portrayal, in case no conditions match
    portrayal = {
        "Shape": "circle",
        "Color": "red",
        "Filled": "true",
        "Layer": 1,
        "r": 0.5
    }

    # Assign portrayal based on agent type
    if isinstance(agent, Sheep):
        portrayal = {"Shape": "circle", "Color": "white", "Filled": "true", "Layer": 1, "r": 0.5}
    elif isinstance(agent, Wolf):
        portrayal = {"Shape": "circle", "Color": "red", "Filled": "true", "Layer": 1, "r": 0.5}
    elif isinstance(agent, Hunter):
        portrayal = {"Shape": "rect", "Color": "black", "Filled": "true", "Layer": 1, "w": 0.8, "h": 0.8}
    elif isinstance(agent, GrassPatch):
        if agent.fully_grown:
            portrayal = {"Shape": "rect",
                         "Color": "green",
                         "Filled": "true",
                         "Layer": 0,
                         "w": 1,
                         "h": 1}
        else:
            portrayal = {"Shape": "rect",
                         "Color": "brown",
                         "Filled": "true",
                         "Layer": 0,
                         "w": 1,
                         "h": 1}
    return portrayal
   """



class Wolf(mesa.Agent):
    def __init__(self,unique_id, model, energy=15, perception_r=3, movement={'step': 2, 'flee': 4},hunger_level=[1, 2, 3, 4, 5], p_reproduction=0.1, energy_gain={'eating_prey': 6}, energy_loss={'step': 1, 'flee': 5}):
        super().__init__(unique_id, model)  # Pass unique_id and model to the parent Agent class
        self.energy = energy
        self.perception_r = perception_r
        self.movement = movement
        self.hunger_level = hunger_level
        self.p_reproduction = p_reproduction
        self.energy_gain = energy_gain
        self.energy_loss = energy_loss

    def step(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} has no valid position during step.")
            return
        # Movement and interaction logic
        self.move()
        self.feed()
        self.reproduce()
        self.check_energy()

    def check_energy(self):
    # Check if the energy level is zero or below
        if self.energy <= 0:
            try:
                # Safely remove the agent from the grid and scheduler
                self.model.increase_deaths_by_energy()
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
            except KeyError as e:
                # Handle cases where the agent is already removed or not found
                print(f"Error while removing agent {self.unique_id}: {e}")



    def feed(self):
        # Find sheep in the same cell
        if self.pos is None:
        # Agent is not on the grid; skip reproduction
            return
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        sheep_list = [sheep for sheep in cell_contents if isinstance(sheep, Sheep)]
        if sheep_list:
            sheep = self.random.choice(sheep_list)
            self.model.grid.remove_agent(sheep)  # Remove the sheep from the grid
            self.model.schedule.remove(sheep)  # Remove the sheep from the scheduler
            self.energy += self.energy_gain['eating_prey']  # Wolf gains energy
            self.model.increase_wolf_kills()
    
    def reproduce(self):
        if self.pos is None:
        # Agent is not on the grid; skip reproduction
            return
        if self.random.random() < self.p_reproduction:
            # Check if the current cell is empty except for this agent
            current_cell = self.model.grid.get_cell_list_contents([self.pos])
            if len(current_cell) == 1:  # Only this wolf is in the cell
                # Create a new wolf in the same cell
                newborn = Wolf(self.model.next_id(), self.model, self.energy, self.perception_radius, self.reproduction_prob)
                self.model.grid.place_agent(newborn, self.pos)
                self.model.schedule.add(newborn)
            
    def move(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return

        # Decide whether to flee or move normally
        if self.should_flee():
            self.flee()
        else:
            self.normal_move()
    def should_flee(self):
        # Make sure the agent has a valid position
        if self.pos is None:
            return False

        # Check for hunters in the perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  # True to include diagonal adjacent cells
            include_center=False,
            radius=self.perception_r
        )

        # Get all agents in these cells
        neighbors = self.model.grid.get_cell_list_contents(neighborhood)

        # Check if any hunter is in the neighborhood
        is_hunter_nearby = any(isinstance(agent, Hunter) for agent in neighbors)
        return is_hunter_nearby
        
    def normal_move(self):
        if self.pos is None:
        # If the agent has no valid position, skip movement
         return
        
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        if not possible_steps:
            # No available steps, agent stays in place
            return
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        self.energy -= self.energy_loss["step"]  # Deduct energy cost for moving

    def flee(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=3  # Fleeing moves the wolf farther
        )
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        self.energy -= self.energy_loss["flee"]  # Deduct energy cost for fleeing

    



class Sheep(mesa.Agent):
    def __init__(self, unique_id,model, energy=10, perception_r=1, movement={'step': 1, 'flee': 2}, p_reproduction=0.2, energy_gain={'eating_grass': 2}, energy_loss={'step': 1, 'flee': 5}):
        super().__init__(unique_id,model)
        self.energy = energy
        self.perception_r = perception_r
        self.movement = movement
        self.p_reproduction = p_reproduction
        self.energy_gain = energy_gain
        self.energy_loss = energy_loss

    def step(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} has no valid position during step.")
            return
        self.move()
        self.feed()
        self.reproduce()
        self.check_energy()

    def check_energy(self):
    # Check if the energy level is zero or below
        if self.energy <= 0:
            try:
                # Safely remove the agent from the grid and scheduler
                self.model.increase_deaths_by_energy()
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
            except KeyError as e:
                # Handle cases where the agent is already removed or not found
                print(f"Error while removing agent {self.unique_id}: {e}")



    def feed(self):
        if self.pos is None:
        # Agent is not on the grid; skip reproduction
            return
        try:
            # Find grass in the same cell
            cell_contents = self.model.grid.get_cell_list_contents([self.pos])
            grass_patches = [grass for grass in cell_contents if isinstance(grass, GrassPatch) and grass.fully_grown]
            if grass_patches:
                grass = self.random.choice(grass_patches)
                grass.fully_grown = False  # The grass is eaten and no longer fully grown
                self.energy += self.energy_gain['eating_grass']  # Sheep gains energy
            else:
                print(f"Sheep {self.unique_id} found no grass to eat.")
        except Exception as e:
            print(f"Error in feeding for Sheep {self.unique_id}: {e}")

    def reproduce(self):
        if self.pos is None:
        # Agent is not on the grid; skip reproduction
            return
        if self.random.random() < self.p_reproduction:
            # Check if the current cell is empty except for this sheep
            current_cell = self.model.grid.get_cell_list_contents([self.pos])
            if len(current_cell) == 1:  # Only this sheep is in the cell
                # Create a new sheep in the same cell
                newborn = Sheep(self.model.next_id(), self.model, self.energy, self.reproduction_prob)
                self.model.grid.place_agent(newborn, self.pos)
                self.model.schedule.add(newborn)

    def move(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return
        # Decide whether to flee or move normally
        if self.should_flee():
            self.flee()
        else:
            self.normal_move()

    def should_flee(self):
    # Make sure the agent has a valid position
        if self.pos is None:
            return False

        # Check for hunters in the perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  # True to include diagonal adjacent cells
            include_center=False,
            radius=self.perception_r
        )

        # Get all agents in these cells
        neighbors = self.model.grid.get_cell_list_contents(neighborhood)

        # Check if any hunter is in the neighborhood
        is_wolf_nearby = any(isinstance(agent, Wolf) for agent in neighbors)
        return is_wolf_nearby
    
    def normal_move(self):
        if self.pos is None:
        # If the agent has no valid position, skip movement
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        if not possible_steps:
            # No available steps, agent stays in place
            return
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        self.energy -= self.energy_loss["step"]  # Deduct energy cost for moving

    def flee(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=2  # Fleeing moves the wolf farther
        )
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        self.energy -= self.energy_loss["flee"]  # Deduct energy cost for fleeing

class Hunter(mesa.Agent):
    def __init__(self, unique_id,model, energy=20, perception_r=3, movement={'step': 1}, weapons={'bullet': 10, 'trap': 3}, energy_loss={'step': 1, 'put_trap': 2}):
        super().__init__(unique_id,model)
        self.energy = energy
        self.perception_r = perception_r
        self.movement = movement
        self.weapons = weapons
        self.energy_loss = energy_loss  
        self.removed = False  # Track if the hunter is currently removed
        self.remove_steps = 0  # Steps since removal 

    def step(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} has no valid position during step.")
            return
        if self.removed:
            self.remove_steps += 1
            if self.remove_steps >= 2:
                # Reintegrate the hunter after being removed for two steps
                self.reintegrate()
        else:
            self.move()
            self.kill_wolf()
            if random.random() < 0.5:
                self.set_trap()
            self.check_energy()

    def check_energy(self):
        if self.energy <= 0:
            self.remove_from_simulation()
            self.model.increase_deaths_by_energy()

    def remove_from_simulation(self):
        # Remove the hunter from active agents and mark as removed
        self.model.grid.remove_agent(self)
        self.removed = True
        self.remove_steps = 0  # Reset the count

    def reintegrate(self):
    # Reintegrate the hunter into the simulation
        self.removed = False
        self.remove_steps = 0
        self.energy = 20  # Reset energy or set to some initial value
        
        # Place the hunter back on the grid at a random valid location
        valid_position = False
        attempts = 0
        while not valid_position and attempts < 10:
            x = self.random.randrange(self.model.grid.width)
            y = self.random.randrange(self.model.grid.height)
            if self.model.grid.is_cell_empty((x, y)):
                valid_position = True
                self.model.grid.place_agent(self, (x, y))
            attempts += 1
        
        if valid_position:
            self.model.schedule.add(self)
        else:
            print(f"Failed to place hunter {self.unique_id} after {attempts} attempts.")

    def move(self):
        # First, scan for nearby wolves within the perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.perception_r
        )
        # Get all agents in these cells
        neighbors = self.model.grid.get_cell_list_contents(neighborhood)
        wolves = [agent for agent in neighbors if isinstance(agent, Wolf)]

        if wolves:
            # Find the closest wolf using a custom distance calculation
            min_distance = float('inf')
            closest_wolf = None
            for wolf in wolves:
                distance = manhattan_distance(self.pos, wolf.pos)  # Replace this with `euclidean_distance` if you prefer
                if distance < min_distance:
                    min_distance = distance
                    closest_wolf = wolf.pos

            # Move towards the closest wolf
            new_position = self.move_towards(closest_wolf)
            self.model.grid.move_agent(self, new_position)
        else:
            # Move randomly if no wolves are detected
            new_position = self.random_move(self.movement['step'])
            self.model.grid.move_agent(self, new_position)

        self.energy -= self.energy_loss['step']

    def random_move(self, radius):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=radius
        )
        return self.random.choice(possible_steps)

    def move_towards(self, target_pos):
        """Move towards a target position if possible."""
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        # Find the step that minimizes the distance to the target
        best_step = None
        min_distance = float('inf')
        for step in possible_steps:
            distance = manhattan_distance(step, target_pos)  # Replace this with `euclidean_distance` if you prefer
            if distance < min_distance:
                min_distance = distance
                best_step = step
        return best_step

    
    def set_trap(self):
        if self.weapons['trap'] > 0:
            # Place a trap in the current cell
            self.model.grid.place_agent(Trap(self.model.next_id(), self.model), self.pos)
            self.weapons['trap'] -= 1
            self.energy -= self.energy_loss['put_trap'] 

    def kill_wolf(self):
        # Check for wolves in the same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        wolves = [agent for agent in cellmates if isinstance(agent, Wolf)]
        if wolves and self.weapons['bullet'] > 0:
            wolf_to_kill = self.random.choice(wolves)
            wolf_to_kill.remove()  # Remove the wolf from the simulation
            self.weapons['bullet'] -= 1  
            self.model.increase_hunter_kills()

class GrassPatch(mesa.Agent):
    """A patch of grass that grows at a fixed rate and it is eaten by sheep."""
    def __init__(self, unique_id, model, regrowth_time, fully_grown=True):
        super().__init__(unique_id, model)
        self.fully_grown = fully_grown
        self.regrowth_time = regrowth_time
        self.time_since_eaten = 0  # Countdown timer to track regrowth

    def step(self):
        # Regrow grass if it's not fully grown
        if not self.fully_grown:
            self.time_since_eaten += 1
            if self.time_since_eaten >= self.regrowth_time:
                self.fully_grown = True
                self.time_since_eaten = 0  # Reset the timer

class Trap(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sprung = False

    def step(self):
        # Check if there are wolves in the same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        wolves = [agent for agent in cellmates if isinstance(agent, Wolf)]
        if wolves:
            # If there is at least one wolf, spring the trap
            self.spring_trap(wolves)

    def spring_trap(self, wolves):
        try:
            # Choose a wolf to be caught by the trap
            if wolves:
                wolf_to_catch = self.random.choice(wolves)
                # Remove the wolf from the simulation
                self.model.grid._remove_agent(self.pos, wolf_to_catch)
                self.model.schedule.remove(wolf_to_catch)
                # Set the trap as sprung
                self.sprung = True
                # Optionally, remove the trap from the grid
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
        except KeyError as e:
            # Handle any issues related to removing wolves that are already gone
            print(f"Error while springing trap at {self.pos}: {e}")


class EcosystemModel(mesa.Model):
    """ A model with some number of agents. """
    def __init__(self, width, height, initial_grass=True, grass_regrowth_time=10):
        # Properly initialize the parent Model class to inherit necessary functionality
        super().__init__()
        
        self.grid = MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True  # Required for the Mesa BatchRunner or Visualization

                # Initialize counters for events
        self.wolf_kills = 0
        self.hunter_kills = 0
        self.deaths_by_energy = 0


        # Create sheep
        for i in range(150):
            sheep = Sheep(self.next_id(), self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(sheep, (x, y))
            if not self.grid.is_cell_empty((x, y)):  # Optional: add a check to confirm placement
                self.schedule.add(sheep)

        # Create wolves
        for i in range(15):
            wolf = Wolf(self.next_id(), self)
          
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(wolf, (x, y))
            if not self.grid.is_cell_empty((x, y)):  # Optional: add a check to confirm placement
                self.schedule.add(wolf)

        # Create hunters
        for i in range(5):
            hunter = Hunter(self.next_id(), self)
       
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(hunter, (x, y))
            if not self.grid.is_cell_empty((x, y)):  # Optional: add a check to confirm placement
                self.schedule.add(hunter)

        if initial_grass:
            for (cell, contents) in self.grid.coord_iter():
                x, y = contents
                fully_grown = self.random.choice([True, False])
                grass = GrassPatch(self.next_id(), self, grass_regrowth_time, fully_grown)
                self.grid.place_agent(grass, (x, y))
                if not self.grid.is_cell_empty((x, y)):  # Optional: add a check to confirm placement
                    self.schedule.add(grass)

        # Create data collector
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: self.count_agents(m, Wolf),
                "Sheep": lambda m: self.count_agents(m, Sheep),
                "Hunters": lambda m: self.count_agents(m, Hunter),
                #"Grass": lambda m: self.count_grass_patches(m)
                "Wolf Kills": lambda m: m.wolf_kills,
                "Hunter Kills": lambda m: m.hunter_kills,
                "Deaths by Energy": lambda m: m.deaths_by_energy,
            }
        )

        # Collect initial data
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # Collect data after every step
        self.datacollector.collect(self)

    @staticmethod
    def count_agents(model, agent_type):
        count = sum(1 for agent in model.schedule.agents if isinstance(agent, agent_type))
        return count

    @staticmethod
    def count_grass_patches(model):
        count = sum(1 for agent in model.schedule.agents if isinstance(agent, GrassPatch) and agent.fully_grown)
        return count
    
    def increase_wolf_kills(self):
        self.wolf_kills += 1

    def increase_hunter_kills(self):
        self.hunter_kills += 1

    def increase_deaths_by_energy(self):
        self.deaths_by_energy += 1
"""
model = EcosystemModel(width=50, height=50, initial_grass=True, grass_regrowth_time=10)
for i in range(100):
    model.step()

# Retrieve the data collected
data = model.datacollector.get_model_vars_dataframe()
print(data)

# You can plot the data using matplotlib or seaborn to visualize trends
import matplotlib.pyplot as plt

data.plot()
plt.xlabel('Step')
plt.ylabel('Count')
plt.title('Population Dynamics')
plt.show()

"""



model_params = {
    "width": 50,
    "height": 50,
    "initial_grass": True,
    "grass_regrowth_time": 10,
}
model = EcosystemModel(width=50, height=50, initial_grass=True, grass_regrowth_time=10)




# Define the Jupyter visualization
page = JupyterViz(
    EcosystemModel,
    model_params,
    measures=["Sheep", "Wolves", "Hunters","Wolf Kills", "Hunter Kills", "Deaths by Energy" ],  # Measures to be tracked and displayed
    name="Wolf-Sheep-Hunter Ecosystem",
    agent_portrayal=agent_portrayal,
)



# This line is required to render the visualization in a Jupyter notebook
