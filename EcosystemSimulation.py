import mesa
import geopandas as gpd
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
import random
from mesa.visualization.modules import CanvasGrid, TextElement
import numpy as np
import datetime

class ShapefileGrid(mesa.space.MultiGrid):
    def __init__(self, width, height, shapefile_path):
        super().__init__(width, height, torus=False)
        self.valid_cells = self.load_valid_cells(shapefile_path)

    def is_cell_occupied(self, pos):
        """
        Checks if a specific cell is occupied by any agent.
        
        Args:
            pos (tuple): The grid coordinates to check (x, y).
        
        Returns:
            bool: True if there is at least one agent in the cell, False otherwise.
        """
        # This method checks if the list of agents at a specific position is empty or not
        return len(self.get_cell_list_contents(pos)) > 0

    def load_valid_cells(self, shapefile_path):
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        # Assuming each row in the GeoDataFrame corresponds to a valid cell area
        valid_cells = set()
        for _, row in gdf.iterrows():
            x, y = row.geometry.centroid.coords[0]  # Assuming the centroids can be considered valid points
            valid_cells.add((int(x), int(y)))  # Convert to grid coordinates if necessary
        return valid_cells
   

      

    def find_empty(self):
        all_cells = list(self.valid_cells)
        empty_cells = [cell for cell in all_cells if not self.is_cell_occupied(cell)]
        if empty_cells:
            return random.choice(empty_cells)
        else:
            raise Exception("No empty cells available within the valid area")
    def get_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two grid positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_within_bounds(self, pos):
        """Checks if a position is within the valid coordinates of the grid."""
        if pos is None:
            return False
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos in model.valid_coords

    
   



class PreyResourceAgent(mesa.Agent):
    def __init__(self, unique_id, model, resource_level):
        super().__init__(unique_id, model)
        self.resource_level = resource_level
        self.resource_level = max(0, self.resource_level + random.randint(-2, 5))

    def step(self):
        pass

class HumanHabitatAgent(mesa.Agent):
    def __init__(self, unique_id, model, human_density):
        super().__init__(unique_id, model)
        self.human_density = human_density

    def step(self):
       pass

class LivestockPresenceAgent(mesa.Agent):
    def __init__(self, unique_id, model, livestock_present=False):
        super().__init__(unique_id, model)
        self.livestock_present = livestock_present

    def step(self):
        pass

class PublicOpinion(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = "Low Concern"
        self.last_state = None  # Track the last state to detect changes

    def step(self):
        # Determine the new state based on the conflict score
        if self.model.conflict_score < 75:
            new_state = "Low Concern"
        elif 50 <= self.model.conflict_score < 130:
            new_state = "Medium Concern"
        else:
            new_state = "High Concern"
        
        # Check if the state has changed
        if new_state != self.last_state:
            self.last_state = new_state  # Update the last state
            self.state = new_state       # Update the current state
            # Communicate the new state to the policy maker
            self.model.policy_maker.receive_update(self.state)
            # Log the state change
            log_communication(self.unique_id, self.model.policy_maker.unique_id, "inform", f"State changed to {self.state}",ontology="Pubblic Opinion Influence", model=self.model)


class PolicyMaker(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_policy = "Normal Operations"
        self.last_policy = None  # Track the last policy to detect changes

    def receive_update(self, opinion_state):
        # Decide on policy based on public opinion
        if opinion_state == "High Concern":
            new_policy = "Decrease Hunter Restrictions"
        elif opinion_state == "Medium Concern":
            new_policy = "Maintain Normal Operations"
        else:
            new_policy = "Increase Hunter Restrictions"
        
        # Check if the policy has changed
        if new_policy != self.last_policy:
            self.last_policy = new_policy  # Update the last policy
            self.current_policy = new_policy  # Update the current policy
            # Communicate the new policy to all hunters
            for agent in self.model.schedule.agents:
                if isinstance(agent, Hunter):
                    agent.update_policy(self.current_policy)
            # Log the policy change
            log_communication(self.unique_id, "all_hunters", "inform", f"Policy changed to {self.current_policy}", ontology="Policy Maker Decision",model=self.model)


class WolfPack(mesa.Agent):
    def __init__(self, unique_id, model, pack_size=5, perception_radius=3, reproduction_prob=0.05):
        super().__init__(unique_id, model)
        self.pack_size = pack_size
        self.perception_radius = perception_radius
        self.reproduction_prob = reproduction_prob

    
    def step(self):
       
        if self.pos is None:
            print(f"Error: Position for WolfPack {self.unique_id} is None")
            return  # Skip the step method if position is None
        # Proceed with the rest of the step metho
        self.check_for_conflicts()
        self.check_pack_size()
        self.reproduce()# Attempt to move towards prey or livestock
        self.check_for_conflicts()
        moved = self.move_towards_prey()
        # If no prey or livestock found, move randomly
        if not moved:
            self.move_randomly()
        self.reproduce()
    
    

    def check_for_conflicts(self):
        """Check the current cell for human or livestock agents and adjust conflict score."""
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        for agent in cell_contents:
            if isinstance(agent, HumanHabitatAgent):
                if agent.human_density > 0.7:  # Example threshold for high density
                    self.model.increase_conflict_score(3)  # Adjust score by 5 for high-density human areas
            elif isinstance(agent, LivestockPresenceAgent) and agent.livestock_present:
                self.model.increase_conflict_score(2)  # Adjust score by 3 for livestock areas

    def move_towards_prey(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return
        # Identify nearby cells with prey or livestock within perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=self.perception_radius
        )
        targets = [cell for cell in neighborhood if any(isinstance(agent, (PreyResourceAgent, LivestockPresenceAgent)) for agent in self.model.grid.get_cell_list_contents(cell)) and self.model.grid.is_within_bounds(cell)]
        
        # Move towards one of the cells with prey/livestock if any are found
        if targets:
            chosen_target = random.choice(targets)
            self.model.grid.move_agent(self, chosen_target)
            return True
        return False

    def move_randomly(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=1  # Assuming random movement is within a single step
        )
        valid_steps = [step for step in possible_steps if self.model.grid.is_within_bounds(step)]
        if valid_steps:
            new_position = random.choice(valid_steps)
            self.model.grid.move_agent(self, new_position)

    def flee(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=2  # Fleeing moves the wolf farther
        )
        valid_steps = [step for step in possible_steps if self.model.grid.is_within_bounds(step)]
        if valid_steps:
            new_position = random.choice(valid_steps)
            self.model.grid.move_agent(self, new_position)
            print(f"WolfPack {self.unique_id} has fled to {new_position} after being attacked.")

    def check_pack_size(self):
        """Check if the pack size exceeds the threshold and split if necessary."""
        if self.pack_size > 20:
            self.split_pack()

    def split_pack(self):
        if self.pack_size > 20:
            new_pack_size = self.pack_size // 2
            self.pack_size = new_pack_size
            new_pack_pos = self.model.grid.find_empty()
            if new_pack_pos:
                new_pack = WolfPack(self.model.next_id(), self.model, new_pack_size)
                self.model.grid.place_agent(new_pack, new_pack_pos)
                self.model.schedule.add(new_pack)

    def reproduce(self):
        # WolfPack reproduction based on the probability and current pack size
        if self.random.random() < self.reproduction_prob:
            self.pack_size += 1

    def remove_if_needed(self):
        # Optionally remove the WolfPack if pack size drops to 0
        if self.pack_size <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

import random

class Hunter(mesa.Agent):
    def __init__(self, unique_id, model, energy=15, respawn_delay=10):
        super().__init__(unique_id, model)
        self.energy = energy
        self.perception_r = 2  # perception radius
        self.movement = {'step': 1}  # movement steps
        self.weapons = {'bullet': 10, 'trap': 3}  # initial weapons
        self.energy_loss = {'step': 1}  # energy lost per action
        self.respawn_delay = respawn_delay
        self.respawn_timer = 0

    def step(self):
        self.move()
        self.kill_wolf()
        
        
            
        self.check_energy()
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            print(f"Respawn in {self.respawn_timer} steps.")
            if self.respawn_timer == 0:
                self.respawn_hunter()

    def update_policy(self, policy):
        if policy == "Increase Hunter Restrictions":
            self.movement['step'] = max(0, self.movement['step'] - 1)  # Decrease movement
        elif policy == "Decrease Hunter Restrictions":
            self.movement['step'] += 1  # Increase movement
        # Log policy update
        #log_communication("Policy Maker", f"Hunter {self.unique_id}", policy)

    def check_energy(self):
        if self.energy <= 0:
            self.respawn_timer = 1
            # Remove the current hunter from the grid and schedule
            if self.pos is not None and self in self.model.grid._grid[self.pos[0]][self.pos[1]]:
                self.model.grid.remove_agent(self)
            if self in self.model.schedule.agents:
                self.model.schedule.remove(self)
            print(f"Hunter {self.unique_id} has died and will be respawned.")

    
    def respawn_hunter(self):
        

        # Create a new hunter to replace the current one
        new_hunter = Hunter(self.model.next_id(), self.model)
        # Attempt to find a random valid position
        attempts = 0
        while attempts < 10:
           
            self.model.place_agent_randomly(new_hunter)
            self.model.schedule.add(new_hunter)
            print(f"Hunter {new_hunter.unique_id} spawned .")
            break
        attempts += 1
        
        if attempts == 10:
            print(f"Failed to find a valid position to respawn Hunter {new_hunter.unique_id}.")

    

    def move(self):
        if self.pos is None:
            print(f"Warning: {self.unique_id} attempted to move but has no valid position.")
            return
        # Scan for nearby Wolf Packs within the perception radius
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=int(self.perception_r)
        )
        neighbors = self.model.grid.get_cell_list_contents(neighborhood)
        wolf_packs = [agent for agent in neighbors if isinstance(agent, WolfPack)]

        if wolf_packs:
            # Find the closest wolf pack using a simple distance calculation
            closest_wolf_pack = min(wolf_packs, key=lambda wp: self.model.grid.get_distance(self.pos, wp.pos))

            # Move towards the closest wolf pack if possible
            new_position = self.move_towards(closest_wolf_pack.pos)
            if self.model.grid.is_within_bounds(new_position):
                self.model.grid.move_agent(self, new_position)
        else:
            # Move randomly if no wolf packs are detected
            new_position = self.random_move(self.movement['step'])
            if new_position and self.model.grid.is_within_bounds(new_position):
                self.model.grid.move_agent(self, new_position)

        self.energy -= 1  # Decrement energy after moving

    def random_move(self, step):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False,
            radius=step
        )
        valid_steps = [step for step in possible_steps if self.model.grid.is_within_bounds(step)]
        if valid_steps:
            return random.choice(valid_steps)
        else:
            print(f"Warning: Agent {self.unique_id} at {self.pos} has no valid moves.")
            return self.pos  # Stay in the same position if no valid moves are available

    def move_towards(self, target_pos):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        valid_steps = [step for step in possible_steps if self.model.grid.is_within_bounds(step)]
        if valid_steps:
            # Find the step that minimizes the distance to the target within the valid steps
            return min(valid_steps, key=lambda step: self.model.grid.get_distance(step, target_pos))
        else:
            return self.pos  # Stay in the same position if no valid moves toward the target
    

    def kill_wolf(self):
        # Locate wolf packs in the same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        wolf_packs = [agent for agent in cellmates if isinstance(agent, WolfPack)]
        if wolf_packs and self.weapons['bullet'] > 0:
            wolf_pack_to_attack = random.choice(wolf_packs)
            if wolf_pack_to_attack.pack_size > 0:
                wolf_pack_to_attack.pack_size -= 1  # Reduce pack size
                self.weapons['bullet'] -= 1  # Use a bullet
                self.model.increase_hunter_kills() 
                print(f"Hunter {self.unique_id} attacked a wolf pack at {self.pos}. Pack size now: {wolf_pack_to_attack.pack_size}")
                wolf_pack_to_attack.flee()  # Trigger fleeing behavior
                if wolf_pack_to_attack.pack_size == 0:
                    self.model.grid.remove_agent(wolf_pack_to_attack)
                    self.model.schedule.remove(wolf_pack_to_attack)
                    print(f"Removed wolf pack at {self.pos} after attack.")
            else:
                print(f"No wolves to attack at {self.pos}.")
        else:
            print(f"No wolf packs or bullets available at {self.pos}.")





class EcosystemModel(mesa.Model):
    def __init__(self, livestock_ratio=0.1,  N_wolfpacks=10, N_hunters=5, num_density_centers=10):
        shapefile_path = '/Users/alexdebertolis/Desktop/MASTestSim/shp/output/GRID.shp'
        self.log_initialized = False
        valid_grid_cells = gpd.read_file(shapefile_path)
        minx, miny, maxx, maxy = valid_grid_cells.total_bounds
        width = int((maxx - minx) // 10000)
        height = int((maxy - miny) // 10000)
        self.grid = ShapefileGrid(width, height,shapefile_path=shapefile_path)
        self.schedule = mesa.time.RandomActivation(self)
        self.current_id = 0
        centroids = valid_grid_cells.geometry.centroid
        self.valid_coords = [(int((p.x - minx) // 10000), int((p.y - miny) // 10000)) for p in centroids]
        self.valid_coords = [(x, y) for x, y in self.valid_coords if 0 <= x < width and 0 <= y < height]
        self.conflict_score= 0
        self.density_map = initialize_human_density(self.grid.width, self.grid.height, num_density_centers, self.valid_coords)
        self.conflict_decrease_interval = 10  # Decrease score every 10 steps
        self.last_decrease_step = 0
        self.public_opinion = PublicOpinion("PO", self)
        self.policy_maker = PolicyMaker("PM", self)
        self.schedule.add(self.public_opinion)
        self.schedule.add(self.policy_maker)


        # Place human agents based on the new density map...
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if (x, y) in self.valid_coords:
                    human_density = self.density_map[x, y]
                    if human_density > 0:  # Only place human habitat where density is non-zero
                        human = HumanHabitatAgent(self.next_id(), self, human_density)
                        self.grid.place_agent(human, (x, y))
                        self.schedule.add(human)


        for coord in self.valid_coords:
            resource_level = random.randint(5, 20)
            prey_agent = PreyResourceAgent(self.next_id(), self, resource_level)
            self.grid.place_agent(prey_agent, coord)
            self.schedule.add(prey_agent)

             # Initialize human habitats based on the density map
        
            if random.random() < livestock_ratio:
                livestock_agent = LivestockPresenceAgent(self.next_id(), self, livestock_present=True)
                self.grid.place_agent(livestock_agent, coord)
                self.schedule.add(livestock_agent)
         # Initialize Wolf Packs
        for i in range(N_wolfpacks):
            pack_size = random.randint(3, 10)  # Random initial pack size
            wolfpack = WolfPack(self.next_id(), self, pack_size)
            
            self.place_agent_randomly(wolfpack)
            self.schedule.add(wolfpack)
        
        # Initialize Hunters
        for i in range(N_hunters):
            hunter = Hunter(self.next_id(), self)
            self.place_agent_randomly(hunter)
            self.schedule.add(hunter)
        
        
        self.hunter_kills = 0

         # Create data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Wolves": lambda m: sum([agent.pack_size for agent in m.schedule.agents if isinstance(agent, WolfPack)]),
                "Wolf Packs": lambda m: sum(1 for agent in m.schedule.agents if isinstance(agent, WolfPack)),  # Count number of WolfPack agents
                "Hunters": lambda m: sum(1 for agent in m.schedule.agents if isinstance(agent, Hunter)),
                "Conflict Score": lambda m: m.conflict_score,
                "Public Opinion": lambda m: m.public_opinion.state,
                "Current Policy": lambda m: m.policy_maker.current_policy
            }
        )
    

    
    
    def place_agent_randomly(self, agent):
        if not self.valid_coords:
            raise ValueError("No valid coordinates available for placement.")
        x, y = random.choice(self.valid_coords)  # Use only valid coordinates
        self.grid.place_agent(agent, (x, y))
        agent.pos = (x, y)  # Explicitly set the position to ensure it's not None
    
    def increase_conflict_score(self, points):
        """Method to increase the conflict score."""
        self.conflict_score += points

    def adjust_conflict_score(self):
        # Decrease the conflict score based on wolves' locations
        wolf_packs = [agent for agent in self.schedule.agents if isinstance(agent, WolfPack)]
        low_conflict_zones = 0

        for wolf in wolf_packs:
            pos = wolf.pos
            cell_contents = self.grid.get_cell_list_contents(pos)
            human_agents = [agent for agent in cell_contents if isinstance(agent, HumanHabitatAgent)]
            livestock_agents = [agent for agent in cell_contents if isinstance(agent, LivestockPresenceAgent)]
            
            # Check if in low conflict area (low human density and no livestock)
            if all(human.human_density < 0.3 for human in human_agents) and not any(livestock.livestock_present for livestock in livestock_agents):
                low_conflict_zones += 1
        
        # If most wolf packs are in low conflict zones, decrease the score
        if low_conflict_zones >= len(wolf_packs) / 2:
            self.conflict_score = max(0, self.conflict_score - 30)  # Or any other suitable decrement
            print(f"Conflict score decreased to {self.conflict_score} due to wolves maintaining distance from conflict areas.")



    

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

        if self.schedule.steps - self.last_decrease_step >= self.conflict_decrease_interval:
            self.adjust_conflict_score()
            self.last_decrease_step = self.schedule.steps
        if self.schedule.steps % 100 == 0:  # Optionally, save every 100 steps       
            data = self.datacollector.get_model_vars_dataframe()
            data.to_csv("simulation_data.csv", index_label="Step")


    def next_id(self):
        self.current_id += 1
        return self.current_id
    
   

   

    def increase_hunter_kills(self):
        self.hunter_kills += 1

    
        


def initialize_human_density(width, height, num_centers, valid_coords):
    density_map = np.zeros((width, height))
    centers = random.sample(valid_coords, num_centers)  # Choose random centers from valid coordinates

    # Parameters for each center, such as its peak density and spread
    for center in centers:
        peak_density = random.uniform(0.6, 1.0)  # Peak density for this center
        spread = random.randint(width//20, width//10)  # Spread factor for Gaussian distribution

        # Apply Gaussian decay for density from the center
        for x in range(width):
            for y in range(height):
                if (x, y) in valid_coords:
                    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    density_map[x, y] += peak_density * np.exp(-((distance**2) / (2 * spread**2)))

    # Normalize the map to ensure densities are within the range [0, 1]
    max_density = np.max(density_map)
    if max_density > 0:
        density_map = density_map / max_density

    return density_map

def log_communication(sender, receiver, act, content, model, language="XML", ontology="simulation_ontology"):
    """
    Logs communication between agents in FIPA ACL format with a timestamp.
    
    Parameters:
        sender (str): The sender of the message.
        receiver (str): The receiver of the message.
        act (str): The communicative act type (e.g., 'inform', 'request').
        content (str): The content of the message.
        model (Model): The simulation model object to check if it's the first log entry.
        language (str): The language the content is encoded in.
        ontology (str): The ontology the content adheres to.
    """
    mode = 'a' if model.log_initialized else 'w'
    model.log_initialized = True  # Ensure subsequent logs are appended.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("simulation_communications.txt", mode) as file:
        acl_message = f"( {act}\n"
        acl_message += f"  sender: {sender}\n"
        acl_message += f"  receiver: {receiver}\n"
        acl_message += f"  language: {language}\n"
        acl_message += f"  ontology: {ontology}\n"
        acl_message += f"  content: {content}\n"
        acl_message += f"  timestamp: {timestamp}\n)"
        file.write(acl_message + "\n\n")
    
def agent_portrayal(agent):
    if isinstance(agent, WolfPack):
        portrayal = {"Shape": "circle", "Color": "green", "Filled": "true", "Layer": 2, "r": 0.8}
        portrayal["text"] = str(agent.pack_size)
        portrayal["text_color"] = "white"
    elif isinstance(agent, Hunter):
        portrayal = {"Shape": "circle", "Color": "red", "Filled": "true", "Layer": 2, "r": 1}
        
        portrayal["text_color"] = "black"
   
    
    else:
        portrayal = {"Shape": "rect", "Color": "#CCCCCC", "Filled": "true", "Layer": 0, "w": 1, "h": 1}
    return portrayal


def get_density_color(density):
    if density > 0.75:
        return "#ff0000"  
    elif density > 0.5:
        return "#ff7765"  
    elif density > 0.25:
        return "#ffc8bd"  
    else:
        return "#CCCCCC"  

def human_portrayal(agent):
    if isinstance(agent, HumanHabitatAgent):
        color = get_density_color(agent.human_density)
        return {
            "Shape": "rect",
            "Color": color,
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1
        }

def livestock_portrayal(agent):
    if isinstance(agent, LivestockPresenceAgent) and agent.livestock_present:
        color = "yellow"  # Yellow for cells with livestock
    else:
        color = "#CCCCCC"  # Gray for cells without livestock

    return {
        "Shape": "rect",
        "Filled": "true",
        "Color": color,
        "Layer": 0,
        "w": 1,
        "h": 1,
        "stroke_color": "black",
        "stroke_width": 0.5
    }


def prey_portrayal(agent):
    if isinstance(agent, PreyResourceAgent):
        color = "brown" if agent.resource_level <= 5 else "lightgreen" if agent.resource_level <= 10 else "green" if agent.resource_level <= 15 else "darkgreen"
        return {
            "Shape": "rect",
            "Filled": "true",
            "Color": color,
            "Layer": 0,
            "w": 1,
            "h": 1,
            "stroke_color": "black",
            "stroke_width": 0.5
        }
    



class LegendElement(TextElement):
    def __init__(self, title, items):
        super().__init__()
        self.title = title
        self.items = items

    def render(self, model):
        legend_html = f"<strong>{self.title}</strong><br>"
        for color, description in self.items:
            legend_html += f'<span style="color:{color};">&#9632;</span> {description}<br>'
        return legend_html
    

ecosystem_legend = LegendElement(
    "Ecosystem Agents",
    [
        ("#008000", "Wolf Pack: Size indicates pack size"),
        ("#ff0000", "Hunter"),
       
    ]
)

# Legends for each grid
human_legend = LegendElement(
    "Human Density Grid",
    [("#ff0000", "High density"), ("#ff7765", "Medium density"), ("#ffc8bd", "Low density")]
)


livestock_legend = LegendElement(
    "Livestock Presence Grid",
    [("#ffff00", "Livestock present"), ("#808080", "No livestock")]
)

prey_legend = LegendElement(
    "Prey Resource Level Grid",
    [("#006400", "High prey level"), ("#008000", "Moderate prey level"), ("#90EE90", "Low prey level"), ("#8B4513", "Very low prey level")]
)

class KillCountElement(TextElement):
    def render(self, model):
        return (
                f"Wolf Killed by Hunters: {model.hunter_kills}<br>"
        )
kill_count_element = KillCountElement()

chart = mesa.visualization.ChartModule([{"Label": "Wolves", "Color": "red"},
                     {"Label": "Hunters", "Color": "blue"}],
                    data_collector_name='datacollector')

class ConflictScoreElement(TextElement):
    def render(self, model):
        return f"Conflict Score: {model.conflict_score}"

# Add to server launch code
conflict_score_element = ConflictScoreElement()


class PolicyAndOpinionElement(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        """
        Generates HTML text representing the current policy and public opinion.
        """
        current_policy = model.policy_maker.current_policy  # Assuming there is a policy_maker agent with a current_policy attribute
        public_opinion_state = model.public_opinion.state  # Assuming there is a public_opinion agent with a state attribute
        html = f"Current Policy: {current_policy}<br>Public Opinion State: {public_opinion_state}"
        return html

policy_and_opinion_element = PolicyAndOpinionElement()

conflict_score_chart = mesa.visualization.ChartModule(
    [{"Label": "Conflict Score", "Color": "Blue"}],
    data_collector_name='datacollector'
)

class WolfPackCountElement(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        """
        Generates HTML text displaying the current number of WolfPack agents.
        """
        wolf_pack_count = sum(1 for agent in model.schedule.agents if isinstance(agent, WolfPack))
        return f"Number of Wolf Packs: {wolf_pack_count}"
pack_count = WolfPackCountElement()
model = EcosystemModel()
vis_width, vis_height = model.grid.width, model.grid.height

# Define text elements for each canvas
grid = CanvasGrid(agent_portrayal, vis_width, vis_height, vis_width * 15, vis_height * 15)
human_canvas = CanvasGrid(human_portrayal, vis_width, vis_height, vis_width * 10, vis_height * 10)
livestock_canvas = CanvasGrid(livestock_portrayal, vis_width, vis_height, vis_width * 10, vis_height * 10)
prey_canvas = CanvasGrid(prey_portrayal, vis_width, vis_height, vis_width * 10, vis_height * 10)
server = ModularServer(
    EcosystemModel,
    [ecosystem_legend,policy_and_opinion_element,grid,human_legend, human_canvas, livestock_legend, livestock_canvas, prey_legend, prey_canvas,chart,kill_count_element,pack_count,conflict_score_element,conflict_score_chart],
    "Ecosystem Grid Visualization",
    {}
)
data = model.datacollector.get_model_vars_dataframe()
data.to_csv("simulation_data.csv", index_label="Step")

server.port = 8515
server.launch(open_browser=True)
