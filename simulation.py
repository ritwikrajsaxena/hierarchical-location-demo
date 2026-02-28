import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import defaultdict, deque

class HierarchicalSimulator:
    """Original simulator class for backward compatibility"""
    def __init__(self, num_regions=4, cities_per_region=5, users_per_city=10,
                 mobility_prob=0.1, call_prob=0.3, forwarding_level=1):
        self.num_regions = num_regions
        self.cities_per_region = cities_per_region
        self.users_per_city = users_per_city
        self.mobility_prob = mobility_prob
        self.call_prob = call_prob
        self.forwarding_level = forwarding_level

        self.G = nx.DiGraph()  # Tree: root -> region -> city -> users
        self.user_locations = {}  # user_id -> city
        self.city_coords = {}  # city -> (lat, lon)
        self._build_hierarchy()
        self._assign_users()
        self._assign_coordinates()

    def _build_hierarchy(self):
        self.G.add_node("Root", level=0)
        for r in range(self.num_regions):
            region = f"Region_{r}"
            self.G.add_node(region, level=1)
            self.G.add_edge("Root", region)
            for c in range(self.cities_per_region):
                city = f"City_{r}_{c}"
                self.G.add_node(city, level=2)
                self.G.add_edge(region, city)

    def _assign_users(self):
        user_id = 0
        for city in [n for n in self.G.nodes if "City" in n]:
            for _ in range(self.users_per_city):
                user = f"User_{user_id}"
                self.G.add_node(user, level=3)
                self.G.add_edge(city, user)
                self.user_locations[user] = city
                user_id += 1

    def _assign_coordinates(self):
        # Approximate USA coordinates (lat/lon) per city
        np.random.seed(42)
        for city in [n for n in self.G.nodes if "City" in n]:
            lat = np.random.uniform(25, 50)
            lon = np.random.uniform(-125, -65)
            self.city_coords[city] = (lat, lon)

    def move_users(self):
        # Users can move to other cities with probability
        for user, city in self.user_locations.items():
            if np.random.rand() < self.mobility_prob:
                new_city = np.random.choice(list(self.city_coords.keys()))
                self.user_locations[user] = new_city

    def simulate_calls(self):
        users = list(self.user_locations.keys())
        calls = []
        for i in range(len(users)):
            if np.random.rand() < self.call_prob:
                j = np.random.randint(len(users))
                if i != j:
                    calls.append((users[i], users[j]))
        return calls

    def compute_latency(self, user1, user2):
        # Compute tree distance from LCA (lowest common ancestor)
        path1 = nx.shortest_path(self.G, source="Root", target=user1)
        path2 = nx.shortest_path(self.G, source="Root", target=user2)
        # Forwarding pointers reduce latency by 'forwarding_level'
        lca_index = 0
        for u, v in zip(path1, path2):
            if u == v:
                lca_index += 1
            else:
                break
        hops = len(path1) + len(path2) - 2*lca_index
        latency = max(hops - self.forwarding_level, 1)
        return latency

    def run_simulation(self, steps=1):
        call_data = []
        for _ in range(steps):
            self.move_users()
            calls = self.simulate_calls()
            for u1, u2 in calls:
                latency = self.compute_latency(u1, u2)
                city1 = self.user_locations[u1]
                city2 = self.user_locations[u2]
                lat1, lon1 = self.city_coords[city1]
                lat2, lon2 = self.city_coords[city2]
                call_data.append({
                    "user1": u1, "user2": u2,
                    "city1": city1, "city2": city2,
                    "lat1": lat1, "lon1": lon1,
                    "lat2": lat2, "lon2": lon2,
                    "latency": latency
                })
        return pd.DataFrame(call_data)


class EnhancedHierarchicalSimulator:
    """Enhanced simulator with multi-level forwarding pointers"""
    def __init__(self, num_regions=4, cities_per_region=5, users_per_city=10,
                 mobility_prob=0.1, call_prob=0.3, max_forwarding_chain=3):
        self.num_regions = num_regions
        self.cities_per_region = cities_per_region
        self.users_per_city = users_per_city
        self.mobility_prob = mobility_prob
        self.call_prob = call_prob
        self.max_forwarding_chain = max_forwarding_chain
        
        # Tree structure
        self.G = nx.DiGraph()
        
        # Location management
        self.user_locations = {}  # user_id -> current_city
        self.user_home_locations = {}  # user_id -> home_city (HLR)
        
        # Forwarding pointers at different levels
        self.forwarding_pointers = {
            'city': defaultdict(list),    # City-level pointers
            'region': defaultdict(list),  # Region-level pointers
            'root': defaultdict(list)     # Root-level pointers
        }
        
        # Performance metrics
        self.metrics = {
            'queries': 0,
            'updates': 0,
            'forwarding_hits': defaultdict(int),
            'latency_history': [],
            'cmr_history': []  # Call-to-Mobility Ratio
        }
        
        # User call patterns for dynamic level adjustment
        self.user_call_frequency = defaultdict(int)
        self.user_mobility_frequency = defaultdict(int)
        
        # Coordinates
        self.city_coords = {}
        
        self._build_hierarchy()
        self._assign_users()
        self._assign_coordinates()
    
    def _build_hierarchy(self):
        """Build the hierarchical tree structure"""
        self.G.add_node("Root", level=0, type='root')
        
        for r in range(self.num_regions):
            region = f"Region_{r}"
            self.G.add_node(region, level=1, type='region', region_id=r)
            self.G.add_edge("Root", region)
            
            for c in range(self.cities_per_region):
                city = f"City_{r}_{c}"
                self.G.add_node(city, level=2, type='city', 
                              region_id=r, city_id=c)
                self.G.add_edge(region, city)
    
    def _assign_users(self):
        """Assign users to cities initially"""
        user_id = 0
        for city in [n for n in self.G.nodes if self.G.nodes[n].get('type') == 'city']:
            for _ in range(self.users_per_city):
                user = f"User_{user_id}"
                self.G.add_node(user, level=3, type='user', home_city=city)
                self.G.add_edge(city, user)
                self.user_locations[user] = city
                self.user_home_locations[user] = city
                user_id += 1
    
    def _assign_coordinates(self):
        """Assign geographic coordinates to cities"""
        np.random.seed(42)
        for city in [n for n in self.G.nodes if self.G.nodes[n].get('type') == 'city']:
            region_id = self.G.nodes[city]['region_id']
            city_id = self.G.nodes[city]['city_id']
            
            # Cluster cities by region
            base_lat = 30 + region_id * 10
            base_lon = -120 + region_id * 15
            
            lat = base_lat + np.random.uniform(-5, 5)
            lon = base_lon + np.random.uniform(-5, 5)
            self.city_coords[city] = (lat, lon)
    
    def move_user_realistic(self, user):
        """Realistic mobility model - users tend to move to nearby cities"""
        current_city = self.user_locations[user]
        current_coords = self.city_coords[current_city]
        
        # Calculate distances to all cities
        cities = list(self.city_coords.keys())
        distances = []
        for city in cities:
            if city != current_city:
                coords = self.city_coords[city]
                dist = np.sqrt((coords[0] - current_coords[0])**2 + 
                             (coords[1] - current_coords[1])**2)
                distances.append(dist)
            else:
                distances.append(float('inf'))
        
        # Probability inversely proportional to distance
        distances = np.array(distances)
        probabilities = 1 / (1 + distances)
        probabilities = probabilities / probabilities.sum()
        
        new_city = np.random.choice(cities, p=probabilities)
        return new_city
    
    def update_location_with_forwarding(self, user, new_city):
        """Update user location and maintain forwarding pointers"""
        old_city = self.user_locations[user]
        
        if old_city == new_city:
            return
        
        # Update metrics
        self.metrics['updates'] += 1
        self.user_mobility_frequency[user] += 1
        
        # Get region information
        old_region = list(self.G.predecessors(old_city))[0]
        new_region = list(self.G.predecessors(new_city))[0]
        
        # Update forwarding pointers based on movement pattern
        if old_region == new_region:
            # Intra-region movement: use city-level forwarding
            self.forwarding_pointers['city'][old_city].append({
                'user': user,
                'new_location': new_city,
                'timestamp': self.metrics['queries']  # Use query count as timestamp
            })
        else:
            # Inter-region movement: use region-level forwarding
            self.forwarding_pointers['region'][old_region].append({
                'user': user,
                'new_location': new_city,
                'timestamp': self.metrics['queries']
            })
            
            # Also update root-level for major movements
            if len(self.forwarding_pointers['region'][old_region]) > self.max_forwarding_chain:
                self.forwarding_pointers['root']['Root'].append({
                    'user': user,
                    'new_location': new_city,
                    'timestamp': self.metrics['queries']
                })
        
        # Limit forwarding chain length
        for level in self.forwarding_pointers:
            for node in self.forwarding_pointers[level]:
                if len(self.forwarding_pointers[level][node]) > self.max_forwarding_chain:
                    # Remove oldest pointer
                    self.forwarding_pointers[level][node].pop(0)
        
        # Update actual location
        self.user_locations[user] = new_city
        
        # Update graph edges
        self.G.remove_edge(old_city, user)
        self.G.add_edge(new_city, user)
    
    def find_user_with_forwarding(self, caller, callee):
        """Find user using forwarding pointers at different levels"""
        latency = 0
        queries = 0
        
        # Start from home location
        home_city = self.user_home_locations[callee]
        current_search_location = home_city
        
        # Check forwarding pointers at different levels
        levels_checked = []
        
        # City-level check
        if home_city in self.forwarding_pointers['city']:
            for pointer in self.forwarding_pointers['city'][home_city]:
                if pointer['user'] == callee:
                    current_search_location = pointer['new_location']
                    self.metrics['forwarding_hits']['city'] += 1
                    levels_checked.append('city')
                    latency += 1
                    queries += 1
                    break
        
        # If not found, check region-level
        if current_search_location == home_city:
            home_region = list(self.G.predecessors(home_city))[0]
            if home_region in self.forwarding_pointers['region']:
                for pointer in self.forwarding_pointers['region'][home_region]:
                    if pointer['user'] == callee:
                        current_search_location = pointer['new_location']
                        self.metrics['forwarding_hits']['region'] += 1
                        levels_checked.append('region')
                        latency += 2
                        queries += 2
                        break
        
        # If still not found, check root-level
        if current_search_location == home_city:
            if 'Root' in self.forwarding_pointers['root']:
                for pointer in self.forwarding_pointers['root']['Root']:
                    if pointer['user'] == callee:
                        current_search_location = pointer['new_location']
                        self.metrics['forwarding_hits']['root'] += 1
                        levels_checked.append('root')
                        latency += 3
                        queries += 3
                        break
        
        # If still not found, do exhaustive search
        if current_search_location != self.user_locations[callee]:
            latency += 5  # Penalty for exhaustive search
            queries += 10
            current_search_location = self.user_locations[callee]
        
        self.metrics['queries'] += queries
        return latency, levels_checked
    
    def compute_optimal_level(self, user):
        """Determine optimal hierarchy level for user based on CMR"""
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves
        
        # Dynamic level adjustment based on CMR
        if cmr > 10:  # High call frequency
            return 'root'  # Keep at higher level for faster access
        elif cmr > 5:
            return 'region'
        else:
            return 'city'  # Keep at lower level for less updates
    
    def simulate_step(self):
        """Simulate one time step"""
        # Move users
        moved_users = []
        for user in list(self.user_locations.keys()):
            if np.random.rand() < self.mobility_prob:
                new_city = self.move_user_realistic(user)
                self.update_location_with_forwarding(user, new_city)
                moved_users.append(user)
        
        # Simulate calls
        users = list(self.user_locations.keys())
        calls = []
        total_latency = 0
        
        for _ in range(int(len(users) * self.call_prob)):
            caller = np.random.choice(users)
            callee = np.random.choice(users)
            
            if caller != callee:
                self.user_call_frequency[caller] += 1
                self.user_call_frequency[callee] += 1
                
                latency, levels = self.find_user_with_forwarding(caller, callee)
                total_latency += latency
                
                calls.append({
                    'caller': caller,
                    'callee': callee,
                    'latency': latency,
                    'forwarding_levels': levels,
                    'optimal_level': self.compute_optimal_level(callee)
                })
        
        # Calculate CMR
        if self.metrics['updates'] > 0:
            cmr = self.metrics['queries'] / self.metrics['updates']
            self.metrics['cmr_history'].append(cmr)
        
        if calls:
            avg_latency = total_latency / len(calls)
            self.metrics['latency_history'].append(avg_latency)
        
        return calls, moved_users
    
    def run_simulation(self, steps=10):
        """Run complete simulation"""
        results = {
            'calls': [],
            'movements': [],
            'forwarding_effectiveness': defaultdict(list)
        }
        
        for step in range(steps):
            calls, moved_users = self.simulate_step()
            
            results['calls'].extend(calls)
            results['movements'].append(len(moved_users))
            
            # Track forwarding effectiveness
            for level in self.metrics['forwarding_hits']:
                results['forwarding_effectiveness'][level].append(
                    self.metrics['forwarding_hits'][level]
                )
        
        return results