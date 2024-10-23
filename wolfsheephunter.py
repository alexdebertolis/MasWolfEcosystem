from matplotlib import pyplot as plt
import mesa
import geopandas as gpd
import random
from shapely.geometry import box
from shapely.ops import unary_union
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import Slider
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from matplotlib import pyplot as plt
from shapely.geometry import Point
from mesa.visualization.modules import TextElement


# Load the shapefile for valid grid cells
shapefile_path = 'shp/output/GRID.shp'  # Path to your valid shapefile
grid = gpd.read_file(shapefile_path)


shapefile_path = '/Users/alexdebertolis/Desktop/MASTestSim/shp/output/GRID.shp'  # Path to your valid shapefile
def visualize_model(model):
    # Get the positions of each type of agent
    hunter_positions = [(agent.pos[0], agent.pos[1]) for agent in model.schedule.agents if isinstance(agent, Hunter) and agent.pos]
    wolf_positions = [(agent.pos[0], agent.pos[1]) for agent in model.schedule.agents if isinstance(agent, Wolf) and agent.pos]
    sheep_positions = [(agent.pos[0], agent.pos[1]) for agent in model.schedule.agents if isinstance(agent, Sheep) and agent.pos]

    # Plotting the shapefile outline
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.boundary.plot(ax=ax, color='black', linewidth=1)

    # Plot hunter positions if they exist
    if hunter_positions:
        hx, hy = zip(*hunter_positions)
        plt.scatter(hx, hy, c='blue', label='Hunters', s=50)

    if wolf_positions:
        wx, wy = zip(*wolf_positions)
        plt.scatter(wx, wy, c='red', label='Wolves', s=50)

    if sheep_positions:
        sx, sy = zip(*sheep_positions)
        plt.scatter(sx, sy, c='green', label='Sheep', s=50)

    # Add titles and labels
    plt.title("Agent Positions in the Grid with Boundaries")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Adding the legend
    plt.legend(loc='upper right', title="Agent Types")

    # Save the figure as a static image
    plt.savefig("agent_positions_with_boundaries.png")

    # Show the plot
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display the plot interactively: {e}")

def manhattan_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

def agent_portrayal(agent):
    
        
    if isinstance(agent, Wolf):
        return {
            "Shape": "circle",
            "Color": "red",
            "Filled": "true",
            "r": 0.5,
            "Layer": 2
        }
    elif isinstance(agent, Sheep):
        return {
            "Shape": "circle",
            "Color": "green",
            "Filled": "true",
            "r": 0.5,
            "Layer": 3
        }
    elif isinstance(agent, Hunter):
        return {
            "Shape": "rect",
            "Color": "blue",
            "Filled": "true",
            "w": 0.6,
            "h": 0.6,
            "Layer": 4
        }
    else: 
    
        return {
            "Shape": "rect",
            "Filled": "true",
            "Color": "#CCCCCC",  # Light gray color for empty cells
            "Layer": 0,
            "w": 1,
            "h": 1,
            "stroke_color": "black",  # Border color for grid cells
            "stroke_width": 0.5       # Border thickness
        }
    
    # No portrayal for grass to avoid clutter
    return None

def grass_portrayal(agent):
    if isinstance(agent, GrassPatch):
        if agent.fully_grown:
            return {
                "Shape": "rect",
                "Color": "green",
                "Filled": "true",
                "Layer": 0,
                "w": 1,
                "h": 1
            }
        else:
            return {
                "Shape": "rect",
                "Color": "brown",
                "Filled": "true",
                "Layer": 0,
                "w": 1,
                "h": 1
            }
    return None



class ShapefileGrid(mesa.space.MultiGrid):
    def __init__(self, valid_cells_gdf, width, height):
        # Initialize the MultiGrid class to inherit the grid behavior
        super().__init__(width, height, torus=False)
        
        self.valid_cells = valid_cells_gdf
        # Calculate grid bounds and set width/height properties for visualization
        self.minx, self.miny, self.maxx, self.maxy = self.valid_cells.total_bounds
        self.width = width
        self.height = height

        # Create a mapping of cell centroids for easier agent placement
        self.centroids = self.valid_cells.geometry.centroid
        self.valid_coords = [(int((p.x - self.minx) // 10000), int((p.y - self.miny) // 10000)) for p in self.centroids]
        # Ensure each position is within the grid bounds
        self.valid_coords = [(x, y) for x, y in self.valid_coords if 0 <= x < self.width and 0 <= y < self.height]

    def get_random_valid_cell(self):
        """Return a random coordinate from the valid cells."""
        return random.choice(self.valid_coords)

    def is_valid_position(self, position):
        """Check if a position is within valid cells."""
        return position in self.valid_coords

    def place_agent(self, agent, position):
        """Place an agent at the given position."""
        if self.is_valid_position(position):
            agent.pos = position
            # Use the inherited `_grid` placement from MultiGrid
            self._grid[position[0]][position[1]].append(agent)
            self._neighborhood_cache.clear()  # Clear neighborhood cache if needed

    def move_agent(self, agent, new_position):
        """Move agent to a new position if valid."""
        if agent.pos is not None:
            # Remove the agent from the current position
            current_pos = agent.pos
            if agent in self._grid[current_pos[0]][current_pos[1]]:
                self._grid[current_pos[0]][current_pos[1]].remove(agent)

        # Update the agent's position and add it to the new position
        if self.is_valid_position(new_position):
            agent.pos = new_position
            self._grid[new_position[0]][new_position[1]].append(agent)




"""
class ShapefileGrid:
    def __init__(self, valid_cells_gdf, width, height):
        self.valid_cells = valid_cells_gdf
        self.width = width
        self.height = height

        # Create a mapping of cell centroids for easier agent placement
        self.centroids = self.valid_cells.geometry.centroid
        self.valid_coords = [(p.x, p.y) for p in self.centroids]

    def get_random_valid_cell(self):
        #Return a random coordinate from the valid cells.
        return random.choice(self.valid_coords)

    def move_agent(self, agent, new_position):
        #Move agent to a new position if valid
            agent.pos = new_position
        else:
            print(f"Warning: Attempted to move agent {agent.unique_id} to an invalid cell {new_position}.")



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
                # Ensure the agent is still in the grid before removing it
                if self in self.model.grid.get_cell_list_contents([self.pos]):
                    # Safely remove the agent from the grid and scheduler
                    if self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
                        self.model.grid.remove_agent(self)
                    else:
                         print(f"Warning: Agent {self.unique_id} not found in position {self.pos} for removal.")
                    self.model.schedule.remove(self)
                    self.model.increase_deaths_by_energy_wolf()
            except KeyError as e:
                # Handle cases where the agent is already removed or not found
                print(f"Error while removing agent {self.unique_id}: {e}")

    def feed(self):
        if self.pos is None:
            # Agent is not on the grid; skip feeding
            return
        
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        sheep_list = [sheep for sheep in cell_contents if isinstance(sheep, Sheep)]
        
        if sheep_list:
            sheep = self.random.choice(sheep_list)
            if sheep in self.model.schedule.agents:
                self.model.grid.remove_agent(sheep)  # Remove the sheep from the grid
                try:
                    self.model.schedule.remove(sheep)  # Remove the sheep from the scheduler
                except KeyError:
                    print(f"Warning: Attempted to remove Sheep {sheep.unique_id} from the schedule, but it was not found.")
            else:
                print(f"Warning: Sheep {sheep.unique_id} was not in the schedule.")
            
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
                newborn = Wolf(self.model.next_id(), self.model, self.energy, self.perception_r, self.p_reproduction)
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
            radius=int(self.perception_r)
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
                # Ensure the agent is still in the grid before removing it
                if self in self.model.grid.get_cell_list_contents([self.pos]):
                    # Safely remove the agent from the grid and scheduler
                    if self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
                        self.model.grid.remove_agent(self)
                    else:
                        print(f"Warning: Agent {self.unique_id} not found in position {self.pos} for removal.")
                    self.model.schedule.remove(self)
                    self.model.increase_deaths_by_energy_sheep()
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
               pass 
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
                newborn = Sheep(self.model.next_id(), self.model, self.energy, self.p_reproduction)
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
            radius=int(self.perception_r)
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
    def __init__(self, unique_id, model, energy=20, perception_r=3, movement={'step': 1}, weapons={'bullet': 10, 'trap': 3}, energy_loss={'step': 1, 'put_trap': 2}):
        super().__init__(unique_id, model)
        self.energy = energy
        self.perception_r = perception_r
        self.movement = movement
        self.weapons = weapons
        self.energy_loss = energy_loss

    def step(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} has no valid position during step.")
            return
        
        self.move()
        self.kill_wolf()
        if random.random() < 0.5:
            self.set_trap()
        self.check_energy()

    def check_energy(self):
        if self.energy <= 0:
            self.respawn_hunter()

    def respawn_hunter(self):
        # Remove the current hunter from the grid and schedule
        if self.pos is not None and self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
            self.model.grid.remove_agent(self)
        if self in self.model.schedule.agents:
            self.model.schedule.remove(self)
        print(f"Hunter {self.unique_id} has died and will be respawned.")

        # Create a new hunter to replace the current one
        new_hunter = Hunter(self.model.next_id(), self.model)
        # Attempt to find a random valid position
        attempts = 0
        while attempts < 10:
            x, y = self.model.grid.get_random_valid_cell()
            if self.model.grid.is_valid_position((x, y)):
                # Place the hunter in a valid cell
                self.model.grid.place_agent(new_hunter, (x, y))
                self.model.schedule.add(new_hunter)
                print(f"Hunter {new_hunter.unique_id} spawned at position {(x, y)}.")
                break
            attempts += 1
        if attempts == 10:
            print(f"Failed to find a valid position to respawn Hunter {new_hunter.unique_id}.")

    def move(self):
        # First, scan for nearby wolves within the perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=int(self.perception_r)
        )
        # Get all agents in these cells
        neighbors = self.model.grid.get_cell_list_contents(neighborhood)
        wolves = [agent for agent in neighbors if isinstance(agent, Wolf)]

        if wolves:
            # Find the closest wolf using a custom distance calculation
            min_distance = float('inf')
            closest_wolf = None
            for wolf in wolves:
                distance = manhattan_distance(self.pos, wolf.pos)
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

        if possible_steps:
            return self.random.choice(possible_steps)
        else:
            # No valid moves are available, so the agent stays in place
            print(f"Warning: Agent {self.unique_id} at {self.pos} has no valid moves.")
            return self.pos  # Stay in the same position

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
            distance = manhattan_distance(step, target_pos)
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
            self.model.grid.remove_agent(wolf_to_kill)  # Remove the wolf from the grid
            self.model.schedule.remove(wolf_to_kill)  # Remove the wolf from the schedule
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
                if self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
                    self.model.grid.remove_agent(self)
                else:
                    print(f"Warning: Agent {self.unique_id} not found in position {self.pos} for removal.")
                self.model.schedule.remove(wolf_to_catch)

                # Set the trap as sprung
                self.sprung = True
                # Optionally, remove the trap from the grid
                if self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
                    self.model.grid.remove_agent(self)
                else:
                    print(f"Warning: Agent {self.unique_id} not found in position {self.pos} for removal.")
                self.model.schedule.remove(self)
                self.model.increase_trap_kills()
        except KeyError as e:
            # Handle any issues related to removing wolves that are already gone
            print(f"Error while springing trap at {self.pos}: {e}")





class EcosystemModel(mesa.Model):
    """ A model with some number of agents. """
    def __init__(self, num_wolves=15, num_sheep=200, num_hunters=5, initial_grass=True, grass_regrowth_time=10):
        # Properly initialize the parent Model class to inherit necessary functionality
        super().__init__()

        # Load the shapefile for valid cells
        valid_grid_cells = gpd.read_file(shapefile_path)
        # Calculate width and height for the visualization based on the bounding box of the shapefile
        minx, miny, maxx, maxy = valid_grid_cells.total_bounds
        width = int((maxx - minx) // 10000)  # Assuming a 10km cell size
        height = int((maxy - miny) // 10000)

        # Create the ShapefileGrid instance
        self.grid = ShapefileGrid(valid_grid_cells, width, height)

        # Initialize agent scheduler
        self.schedule = RandomActivation(self)

        # Initialize counters for events
        self.wolf_kills = 0
        self.hunter_kills = 0
        self.deaths_by_energy_wolves = 0
        self.deaths_by_energy_sheep = 0
        self.trap_kills = 0

        # Create sheep
        for i in range(num_sheep):
            sheep = Sheep(self.next_id(), self)
            x, y = self.grid.get_random_valid_cell()
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        # Create wolves
        for i in range(num_wolves):
            wolf = Wolf(self.next_id(), self)
            x, y = self.grid.get_random_valid_cell()
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        # Create hunters
        for i in range(num_hunters):
            hunter = Hunter(self.next_id(), self)
            x, y = self.grid.get_random_valid_cell()
            self.grid.place_agent(hunter, (x, y))
            self.schedule.add(hunter)

        if initial_grass:
            for i, position in enumerate(self.grid.valid_coords):
                fully_grown = self.random.choice([True, False])
                grass = GrassPatch(self.next_id(), self, grass_regrowth_time, fully_grown)
                self.grid.place_agent(grass, position)
                self.schedule.add(grass)

        # Create data collector
        self.datacollector = mesa.DataCollector(
            {
                "Wolves": lambda m: self.count_agents(m, Wolf),
                "Sheep": lambda m: self.count_agents(m, Sheep),
                "Hunters": lambda m: self.count_agents(m, Hunter),
                "Wolf Kills": lambda m: m.wolf_kills,
                "Hunter Kills": lambda m: m.hunter_kills,
                "Deaths by Energy (Wolves)": lambda m: m.deaths_by_energy_wolves,
                "Deaths by Energy (Sheep)": lambda m: m.deaths_by_energy_sheep,
                "Trap Kills": lambda m: m.trap_kills,
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

    def increase_wolf_kills(self):
        self.wolf_kills += 1

    def increase_hunter_kills(self):
        self.hunter_kills += 1

    
    def increase_deaths_by_energy_wolf(self):
        self.deaths_by_energy_wolves += 1

    def increase_deaths_by_energy_sheep(self):
        self.deaths_by_energy_sheep += 1

    def increase_trap_kills(self):
        self.trap_kills += 1
    def export_data(self, filename='model_data.csv'):
        """Export the collected data to a CSV file."""
        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv(filename)
        print(f"Data successfully exported to {filename}")

"""
model = EcosystemModel( initial_grass=True, grass_regrowth_time=10)
for i in range(100):
    model.step()
    visualize_model(model)

# Retrieve the data collected
data = model.datacollector.get_model_vars_dataframe()
print(data)
data.plot()
plt.xlabel('Step')
plt.ylabel('Count')
plt.title('Population Dynamics')
plt.show()
"""
class KillCountElement(TextElement):
    def render(self, model):
        return (f"Preys Killed by Wolves: {model.wolf_kills}<br>"
                f"Wolf Killed by Hunters: {model.hunter_kills}<br>"
                f"Deaths by Energy (Wolves): {model.deaths_by_energy_wolves}<br>"
                f"Deaths by Energy (Sheep): {model.deaths_by_energy_sheep}<br>"
                f"Wolves Killed by Traps: {model.trap_kills}")
kill_count_element = KillCountElement()

model = EcosystemModel(num_wolves=15, num_sheep=200, num_hunters=5, initial_grass=True, grass_regrowth_time=10)

# Get width and height from model grid
vis_width = model.grid.width
vis_height = model.grid.height

# Create the canvas grid
canvas_element = CanvasGrid(agent_portrayal, vis_width, vis_height, vis_width * 15, vis_height * 15)
canvas_element_grass = CanvasGrid(grass_portrayal, vis_width, vis_height, vis_width * 10, vis_height * 10)
# Create the chart module
chart = ChartModule([{"Label": "Wolves", "Color": "red"},
                     {"Label": "Sheep", "Color": "green"},
                     {"Label": "Hunters", "Color": "blue"}],
                    data_collector_name='datacollector')

# CanvasGrid setup


# Set up the server
server = ModularServer(
    EcosystemModel,
    [canvas_element,canvas_element_grass, chart, kill_count_element],
    "Wolf-Sheep-Hunter Ecosystem",
    {"num_wolves": 15,
     "num_sheep": 150,
     "num_hunters": 5,
     "initial_grass": True,
     "grass_regrowth_time": 10}
)



server.port = 8511  # Set port for visualization



server.launch(open_browser=True)

