import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import defaultdict, deque
import math

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

        self.G = nx.DiGraph()
        self.user_locations = {}
        self.city_coords = {}
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
        np.random.seed(42)
        for city in [n for n in self.G.nodes if "City" in n]:
            lat = np.random.uniform(25, 50)
            lon = np.random.uniform(-125, -65)
            self.city_coords[city] = (lat, lon)

    def move_users(self):
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
        path1 = nx.shortest_path(self.G, source="Root", target=user1)
        path2 = nx.shortest_path(self.G, source="Root", target=user2)
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
    """Enhanced simulator with multi-level forwarding pointers and replication"""
    def __init__(self, num_regions=4, cities_per_region=5, users_per_city=10,
                 mobility_prob=0.1, call_prob=0.3, max_forwarding_chain=3,
                 enable_replication=False, replication_threshold=5, 
                 max_replicas=3, replication_strategy='CMR-based'):
        
        # Basic parameters
        self.num_regions = num_regions
        self.cities_per_region = cities_per_region
        self.users_per_city = users_per_city
        self.mobility_prob = mobility_prob
        self.call_prob = call_prob
        self.max_forwarding_chain = max_forwarding_chain
        
        # Replication parameters
        self.enable_replication = enable_replication
        self.replication_threshold = replication_threshold
        self.max_replicas = max_replicas
        self.replication_strategy = replication_strategy
        
        # Tree structure
        self.G = nx.DiGraph()
        
        # Location management
        self.user_locations = {}
        self.user_home_locations = {}
        
        # Forwarding pointers
        self.forwarding_pointers = {
            'city': defaultdict(list),
            'region': defaultdict(list),
            'root': defaultdict(list)
        }
        
        # Replication data structures
        self.replica_locations = defaultdict(set)  # user -> set of replica nodes
        self.access_frequency_matrix = defaultdict(lambda: defaultdict(int))  # user -> node -> count
        self.replica_access_history = defaultdict(list)  # Track replica usage
        
        # Cost tracking
        self.costs = {
            'search_without_replication': 0,
            'search_with_replication': 0,
            'update_without_replication': 0,
            'update_with_replication': 0,
            'storage_cost': 0,
            'consistency_maintenance': 0,
            'search_costs_per_step': [],
            'update_costs_per_step': [],
            'replication_decisions': []
        }
        
        # Performance metrics
        self.metrics = {
            'queries': 0,
            'updates': 0,
            'forwarding_hits': defaultdict(int),
            'replica_hits': 0,
            'latency_history': [],
            'cmr_history': [],
            'replication_benefit': [],
            'replica_count_history': []
        }
        
        # User patterns
        self.user_call_frequency = defaultdict(int)
        self.user_mobility_frequency = defaultdict(int)
        self.user_call_sources = defaultdict(lambda: defaultdict(int))  # user -> source_node -> count
        
        # Coordinates
        self.city_coords = {}
        
        # Build network
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
            
            base_lat = 30 + region_id * 10
            base_lon = -120 + region_id * 15
            
            lat = base_lat + np.random.uniform(-5, 5)
            lon = base_lon + np.random.uniform(-5, 5)
            self.city_coords[city] = (lat, lon)
    
    def calculate_replication_benefit(self, user, node):
        """Calculate the benefit of replicating user's data at a node"""
        # Get access frequency from this node
        access_freq = self.access_frequency_matrix[user][node]
        
        # Get user's CMR
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves
        
        # Calculate costs
        search_cost_saved = access_freq * 2  # Assume 2 hops saved per local access
        update_cost_added = moves * 1  # Cost of maintaining replica
        
        # Benefit calculation
        benefit = search_cost_saved - update_cost_added
        
        return benefit, cmr
    
    def update_replicas(self, user):
        """Update replica placement based on access patterns"""
        if not self.enable_replication:
            return
        
        # Get all nodes that have accessed this user
        access_nodes = self.user_call_sources[user]
        
        # Calculate benefits for each node
        replica_candidates = []
        for node, freq in access_nodes.items():
            if node != self.user_locations[user]:  # Don't replicate at current location
                benefit, cmr = self.calculate_replication_benefit(user, node)
                
                if self.replication_strategy == 'CMR-based':
                    if cmr >= self.replication_threshold:
                        replica_candidates.append((node, benefit, freq))
                elif self.replication_strategy == 'Access-frequency':
                    if freq >= self.replication_threshold:
                        replica_candidates.append((node, benefit, freq))
                elif self.replication_strategy == 'Hybrid':
                    if cmr >= self.replication_threshold or freq >= self.replication_threshold * 2:
                        replica_candidates.append((node, benefit, freq))
        
        # Sort by benefit and select top candidates
        replica_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Clear old replicas
        old_replicas = len(self.replica_locations[user])
        self.replica_locations[user].clear()
        
        # Add new replicas (up to max_replicas)
        for i, (node, benefit, freq) in enumerate(replica_candidates[:self.max_replicas]):
            if benefit > 0:  # Only replicate if beneficial
                self.replica_locations[user].add(node)
        
        # Update storage cost
        new_replicas = len(self.replica_locations[user])
        self.costs['storage_cost'] += new_replicas
        
        # Track replication decision
        self.costs['replication_decisions'].append({
            'user': user,
            'old_count': old_replicas,
            'new_count': new_replicas,
            'locations': list(self.replica_locations[user])
        })
    
    def find_user_with_replication(self, caller, callee):
        """Find user using replicas and forwarding pointers"""
        caller_city = self.user_locations[caller]
        callee_city = self.user_locations[callee]
        
        # Track access for replication decisions
        self.user_call_sources[callee][caller_city] += 1
        self.access_frequency_matrix[callee][caller_city] += 1
        
        # Check if replica exists at caller's location
        if self.enable_replication and caller_city in self.replica_locations[callee]:
            # Local replica hit!
            self.metrics['replica_hits'] += 1
            self.costs['search_with_replication'] += 1  # Minimal cost for local access
            return 1, ['replica']
        
        # Otherwise, use standard forwarding pointer search
        latency, levels = self.find_user_with_forwarding(caller, callee)
        
        if self.enable_replication:
            self.costs['search_with_replication'] += latency
        else:
            self.costs['search_without_replication'] += latency
        
        return latency, levels
    
    def move_user_realistic(self, user):
        """Realistic mobility model"""
        current_city = self.user_locations[user]
        current_coords = self.city_coords[current_city]
        
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
        
        distances = np.array(distances)
        probabilities = 1 / (1 + distances)
        probabilities = probabilities / probabilities.sum()
        
        new_city = np.random.choice(cities, p=probabilities)
        return new_city
    
    def update_location_with_forwarding(self, user, new_city):
        """Update location with forwarding pointers and replica management"""
        old_city = self.user_locations[user]
        
        if old_city == new_city:
            return
        
        # Update metrics
        self.metrics['updates'] += 1
        self.user_mobility_frequency[user] += 1
        
        # Calculate update costs
        base_update_cost = 1
        
        # Cost for updating forwarding pointers
        self.costs['update_without_replication'] += base_update_cost
        
        # Additional cost for updating replicas
        if self.enable_replication:
            replica_update_cost = len(self.replica_locations[user]) * 0.5
            self.costs['update_with_replication'] += base_update_cost + replica_update_cost
            self.costs['consistency_maintenance'] += replica_update_cost
        
        # Get region information
        old_region = list(self.G.predecessors(old_city))[0]
        new_region = list(self.G.predecessors(new_city))[0]
        
        # Update forwarding pointers
        if old_region == new_region:
            self.forwarding_pointers['city'][old_city].append({
                'user': user,
                'new_location': new_city,
                'timestamp': self.metrics['queries']
            })
        else:
            self.forwarding_pointers['region'][old_region].append({
                'user': user,
                'new_location': new_city,
                'timestamp': self.metrics['queries']
            })
            
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
                    self.forwarding_pointers[level][node].pop(0)
        
        # Update actual location
        self.user_locations[user] = new_city
        
        # Update graph edges
        self.G.remove_edge(old_city, user)
        self.G.add_edge(new_city, user)
        
        # Update replicas if enabled
        if self.enable_replication:
            self.update_replicas(user)
    
    def find_user_with_forwarding(self, caller, callee):
        """Find user using forwarding pointers"""
        latency = 0
        queries = 0
        
        home_city = self.user_home_locations[callee]
        current_search_location = home_city
        
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
        
        # Region-level check
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
        
        # Root-level check
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
        
        # Exhaustive search if needed
        if current_search_location != self.user_locations[callee]:
            latency += 5
            queries += 10
            current_search_location = self.user_locations[callee]
        
        self.metrics['queries'] += queries
        return latency, levels_checked
    
    def compute_optimal_level(self, user):
        """Determine optimal hierarchy level for user based on CMR"""
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves
        
        if cmr > 10:
            return 'root'
        elif cmr > 5:
            return 'region'
        else:
            return 'city'
    
    def simulate_step(self):
        """Simulate one time step"""
        step_search_cost = 0
        step_update_cost = 0
        
        # Move users
        moved_users = []
        for user in list(self.user_locations.keys()):
            if np.random.rand() < self.mobility_prob:
                new_city = self.move_user_realistic(user)
                self.update_location_with_forwarding(user, new_city)
                moved_users.append(user)
                step_update_cost += 1
        
        # Update replicas periodically for all active users
        if self.enable_replication:
            active_users = set()
            for user in self.user_call_frequency:
                if self.user_call_frequency[user] > 0:
                    active_users.add(user)
                    self.update_replicas(user)
        
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
                
                latency, levels = self.find_user_with_replication(caller, callee)
                total_latency += latency
                step_search_cost += latency
                
                calls.append({
                    'caller': caller,
                    'callee': callee,
                    'latency': latency,
                    'forwarding_levels': levels,
                    'optimal_level': self.compute_optimal_level(callee),
                    'used_replica': 'replica' in levels
                })
        
        # Track costs per step
        self.costs['search_costs_per_step'].append(step_search_cost)
        self.costs['update_costs_per_step'].append(step_update_cost)
        
        # Calculate CMR
        if self.metrics['updates'] > 0:
            cmr = self.metrics['queries'] / self.metrics['updates']
            self.metrics['cmr_history'].append(cmr)
        
        # Track average latency
        if calls:
            avg_latency = total_latency / len(calls)
            self.metrics['latency_history'].append(avg_latency)
        
        # Track replica count
        total_replicas = sum(len(replicas) for replicas in self.replica_locations.values())
        self.metrics['replica_count_history'].append(total_replicas)
        
        # Calculate replication benefit
        if self.enable_replication:
            benefit = (self.costs['search_without_replication'] - self.costs['search_with_replication']) - \
                     (self.costs['update_with_replication'] - self.costs['update_without_replication'])
            self.metrics['replication_benefit'].append(benefit)
        
        return calls, moved_users
    
    def run_simulation(self, steps=10):
        """Run complete simulation"""
        results = {
            'calls': [],
            'movements': [],
            'forwarding_effectiveness': defaultdict(list),
            'replication_stats': [],
            'cost_comparison': {}
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
            
            # Track replication statistics
            if self.enable_replication:
                total_replicas = sum(len(replicas) for replicas in self.replica_locations.values())
                results['replication_stats'].append({
                    'step': step,
                    'total_replicas': total_replicas,
                    'replica_hits': self.metrics['replica_hits'],
                    'users_with_replicas': len([u for u, r in self.replica_locations.items() if len(r) > 0])
                })
        
        # Final cost comparison
        results['cost_comparison'] = {
            'search_without_replication': self.costs['search_without_replication'],
            'search_with_replication': self.costs['search_with_replication'],
            'update_without_replication': self.costs['update_without_replication'],
            'update_with_replication': self.costs['update_with_replication'],
            'storage_cost': self.costs['storage_cost'],
            'consistency_cost': self.costs['consistency_maintenance']
        }
        
        return results
    
    def get_replication_analysis(self):
        """Get detailed replication analysis"""
        analysis = {
            'users_with_replicas': {},
            'replica_distribution': defaultdict(int),
            'access_patterns': {},
            'cmr_analysis': {}
        }
        
        # Analyze each user's replication status
        for user in self.user_locations:
            replicas = self.replica_locations[user]
            if replicas:
                analysis['users_with_replicas'][user] = {
                    'locations': list(replicas),
                    'count': len(replicas),
                    'cmr': self.user_call_frequency[user] / max(self.user_mobility_frequency[user], 1)
                }
            
            # Replica distribution
            analysis['replica_distribution'][len(replicas)] += 1
        
        # Access patterns
        for user, sources in self.user_call_sources.items():
            if sources:
                analysis['access_patterns'][user] = dict(sources)
        
        # CMR-based grouping
        for user in self.user_locations:
            calls = self.user_call_frequency[user]
            moves = max(self.user_mobility_frequency[user], 1)
            cmr = calls / moves
            
            if cmr > 10:
                category = 'high_cmr'
            elif cmr > 5:
                category = 'medium_cmr'
            else:
                category = 'low_cmr'
            
            if category not in analysis['cmr_analysis']:
                analysis['cmr_analysis'][category] = []
            
            analysis['cmr_analysis'][category].append({
                'user': user,
                'cmr': cmr,
                'replicas': len(self.replica_locations[user])
            })
        
        return analysis