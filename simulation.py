import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import defaultdict, deque
import math

# USA boundary coordinates
USA_BOUNDS = {
    'lat_min': 25.0,
    'lat_max': 48.0,
    'lon_min': -124.0,
    'lon_max': -67.0
}

# Major US regions with realistic coordinates
US_REGIONS = {
    0: {'name': 'West', 'lat_center': 40.0, 'lon_center': -115.0,
        'lat_range': 8, 'lon_range': 10},
    1: {'name': 'Midwest', 'lat_center': 41.0, 'lon_center': -90.0,
        'lat_range': 6, 'lon_range': 8},
    2: {'name': 'South', 'lat_center': 33.0, 'lon_center': -85.0,
        'lat_range': 6, 'lon_range': 10},
    3: {'name': 'Northeast', 'lat_center': 42.0, 'lon_center': -73.0,
        'lat_range': 4, 'lon_range': 6},
    4: {'name': 'Southwest', 'lat_center': 35.0, 'lon_center': -105.0,
        'lat_range': 5, 'lon_range': 8},
    5: {'name': 'Southeast', 'lat_center': 30.0, 'lon_center': -82.0,
        'lat_range': 5, 'lon_range': 6},
    6: {'name': 'Central', 'lat_center': 38.0, 'lon_center': -98.0,
        'lat_range': 6, 'lon_range': 8},
    7: {'name': 'Pacific', 'lat_center': 45.0, 'lon_center': -122.0,
        'lat_range': 5, 'lon_range': 5}
}


def is_valid_us_coordinate(lat, lon):
    """Check if coordinate is within continental US and not in Great Lakes"""
    if lat < USA_BOUNDS['lat_min'] or lat > USA_BOUNDS['lat_max']:
        return False
    if lon < USA_BOUNDS['lon_min'] or lon > USA_BOUNDS['lon_max']:
        return False
    if 41 < lat < 47 and -92 < lon < -76:
        if 42 < lat < 46 and -88 < lon < -78:
            return False
    if lon > -75 and lat < 35:
        if lon > -78:
            return False
    return True


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
            region_id = int(city.split('_')[1])
            region_info = US_REGIONS[region_id % len(US_REGIONS)]
            attempts = 0
            while attempts < 50:
                lat = region_info['lat_center'] + np.random.uniform(
                    -region_info['lat_range'] / 2, region_info['lat_range'] / 2)
                lon = region_info['lon_center'] + np.random.uniform(
                    -region_info['lon_range'] / 2, region_info['lon_range'] / 2)
                if is_valid_us_coordinate(lat, lon):
                    break
                attempts += 1
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
        hops = len(path1) + len(path2) - 2 * lca_index
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
    """Enhanced simulator with multi-level forwarding pointers and replication.

    Key fix: both 'with replication' and 'without replication' latencies are
    computed for every single call *inside the same simulation run* so the
    comparison is always apples-to-apples (same movements, same calls).
    """

    def __init__(self, num_regions=4, cities_per_region=5, users_per_city=10,
                 mobility_prob=0.1, call_prob=0.3, max_forwarding_chain=3,
                 enable_replication=False, replication_threshold=5,
                 max_replicas=3, replication_strategy='CMR-based'):
        self.num_regions = min(num_regions, len(US_REGIONS))
        self.cities_per_region = cities_per_region
        self.users_per_city = users_per_city
        self.mobility_prob = mobility_prob
        self.call_prob = call_prob
        self.max_forwarding_chain = max_forwarding_chain
        self.enable_replication = enable_replication
        self.replication_threshold = replication_threshold
        self.max_replicas = max_replicas
        self.replication_strategy = replication_strategy
        self.G = nx.DiGraph()
        self.user_locations = {}
        self.user_home_locations = {}
        self.forwarding_pointers = {
            'city': defaultdict(list),
            'region': defaultdict(list),
            'root': defaultdict(list)
        }
        self.replica_locations = defaultdict(set)
        self.access_frequency_matrix = defaultdict(lambda: defaultdict(int))
        self.replica_access_history = defaultdict(list)
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
        self.metrics = {
            'queries': 0,
            'updates': 0,
            'forwarding_hits': defaultdict(int),
            'replica_hits': 0,
            'latency_history': [],
            'latency_no_repl_history': [],
            'cmr_history': [],
            'replication_benefit': [],
            'replica_count_history': []
        }
        self.user_call_frequency = defaultdict(int)
        self.user_mobility_frequency = defaultdict(int)
        self.user_call_sources = defaultdict(lambda: defaultdict(int))
        self.city_coords = {}
        self._build_hierarchy()
        self._assign_users()
        self._assign_coordinates()

    # ------------------------------------------------------------------
    # HIERARCHY SETUP
    # ------------------------------------------------------------------
    def _build_hierarchy(self):
        self.G.add_node("Root", level=0, type='root')
        for r in range(self.num_regions):
            region_name = US_REGIONS[r]['name']
            region = f"Region_{r}_{region_name}"
            self.G.add_node(region, level=1, type='region', region_id=r)
            self.G.add_edge("Root", region)
            for c in range(self.cities_per_region):
                city = f"City_{r}_{c}"
                self.G.add_node(city, level=2, type='city',
                                region_id=r, city_id=c)
                self.G.add_edge(region, city)

    def _assign_users(self):
        user_id = 0
        for city in [n for n in self.G.nodes
                     if self.G.nodes[n].get('type') == 'city']:
            for _ in range(self.users_per_city):
                user = f"User_{user_id}"
                self.G.add_node(user, level=3, type='user', home_city=city)
                self.G.add_edge(city, user)
                self.user_locations[user] = city
                self.user_home_locations[user] = city
                user_id += 1

    def _assign_coordinates(self):
        np.random.seed(42)
        for city in [n for n in self.G.nodes
                     if self.G.nodes[n].get('type') == 'city']:
            region_id = self.G.nodes[city]['region_id']
            region_info = US_REGIONS[region_id]
            attempts = 0
            while attempts < 50:
                lat = region_info['lat_center'] + np.random.uniform(
                    -region_info['lat_range'] / 2,
                    region_info['lat_range'] / 2)
                lon = region_info['lon_center'] + np.random.uniform(
                    -region_info['lon_range'] / 2,
                    region_info['lon_range'] / 2)
                if is_valid_us_coordinate(lat, lon):
                    lat += np.random.uniform(-0.5, 0.5)
                    lon += np.random.uniform(-0.5, 0.5)
                    break
                attempts += 1
            if attempts >= 50:
                lat = region_info['lat_center'] + np.random.uniform(-1, 1)
                lon = region_info['lon_center'] + np.random.uniform(-1, 1)
            self.city_coords[city] = (lat, lon)

    # ------------------------------------------------------------------
    # REPLICATION
    # ------------------------------------------------------------------
    def calculate_replication_benefit(self, user, node):
        access_freq = self.access_frequency_matrix[user][node]
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves
        search_cost_saved = access_freq * 2
        update_cost_added = moves * 1
        benefit = search_cost_saved - update_cost_added
        return benefit, cmr

    def update_replicas(self, user):
        if not self.enable_replication:
            return
        access_nodes = self.user_call_sources[user]
        if not access_nodes:
            return
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves

        replica_candidates = []
        for node, freq in access_nodes.items():
            if node != self.user_locations[user]:
                benefit, user_cmr = self.calculate_replication_benefit(
                    user, node)
                if self.replication_strategy == 'CMR-based':
                    if user_cmr >= self.replication_threshold:
                        replica_candidates.append((node, benefit, freq))
                elif self.replication_strategy == 'Access-frequency':
                    if freq >= self.replication_threshold:
                        replica_candidates.append((node, benefit, freq))
                elif self.replication_strategy == 'Hybrid':
                    if (user_cmr >= self.replication_threshold
                            or freq >= max(self.replication_threshold // 2, 1)):
                        replica_candidates.append((node, benefit, freq))

        replica_candidates.sort(key=lambda x: x[1], reverse=True)
        old_replicas = len(self.replica_locations[user])
        self.replica_locations[user].clear()

        for node, benefit, freq in replica_candidates[:self.max_replicas]:
            self.replica_locations[user].add(node)

        new_replicas = len(self.replica_locations[user])
        self.costs['storage_cost'] += new_replicas
        self.costs['replication_decisions'].append({
            'user': user,
            'old_count': old_replicas,
            'new_count': new_replicas,
            'locations': list(self.replica_locations[user]),
            'cmr': cmr
        })

    # ------------------------------------------------------------------
    # LATENCY – forwarding-pointer-only (baseline)
    # ------------------------------------------------------------------
    def _forwarding_latency(self, caller, callee):
        """Return the latency using ONLY forwarding pointers (no replicas).

        This method does **not** mutate any metrics so it can be called as a
        side-effect-free comparison alongside the replication-aware path.
        """
        home_city = self.user_home_locations[callee]
        current = home_city
        latency = 0

        # City-level
        if home_city in self.forwarding_pointers['city']:
            for ptr in self.forwarding_pointers['city'][home_city]:
                if ptr['user'] == callee:
                    current = ptr['new_location']
                    latency += 1
                    break

        # Region-level
        if current == home_city:
            home_region = list(self.G.predecessors(home_city))[0]
            if home_region in self.forwarding_pointers['region']:
                for ptr in self.forwarding_pointers['region'][home_region]:
                    if ptr['user'] == callee:
                        current = ptr['new_location']
                        latency += 2
                        break

        # Root-level
        if current == home_city:
            if 'Root' in self.forwarding_pointers['root']:
                for ptr in self.forwarding_pointers['root']['Root']:
                    if ptr['user'] == callee:
                        current = ptr['new_location']
                        latency += 3
                        break

        # Exhaustive
        if current != self.user_locations[callee]:
            latency += 5

        return latency

    # ------------------------------------------------------------------
    # LATENCY – with forwarding pointers (mutating version for metrics)
    # ------------------------------------------------------------------
    def find_user_with_forwarding(self, caller, callee):
        latency = 0
        queries = 0
        home_city = self.user_home_locations[callee]
        current = home_city
        levels_checked = []

        if home_city in self.forwarding_pointers['city']:
            for ptr in self.forwarding_pointers['city'][home_city]:
                if ptr['user'] == callee:
                    current = ptr['new_location']
                    self.metrics['forwarding_hits']['city'] += 1
                    levels_checked.append('city')
                    latency += 1
                    queries += 1
                    break

        if current == home_city:
            home_region = list(self.G.predecessors(home_city))[0]
            if home_region in self.forwarding_pointers['region']:
                for ptr in self.forwarding_pointers['region'][home_region]:
                    if ptr['user'] == callee:
                        current = ptr['new_location']
                        self.metrics['forwarding_hits']['region'] += 1
                        levels_checked.append('region')
                        latency += 2
                        queries += 2
                        break

        if current == home_city:
            if 'Root' in self.forwarding_pointers['root']:
                for ptr in self.forwarding_pointers['root']['Root']:
                    if ptr['user'] == callee:
                        current = ptr['new_location']
                        self.metrics['forwarding_hits']['root'] += 1
                        levels_checked.append('root')
                        latency += 3
                        queries += 3
                        break

        if current != self.user_locations[callee]:
            latency += 5
            queries += 10

        self.metrics['queries'] += queries
        return latency, levels_checked

    # ------------------------------------------------------------------
    # LATENCY – replication-aware wrapper
    # ------------------------------------------------------------------
    def find_user_with_replication(self, caller, callee):
        caller_city = self.user_locations[caller]

        self.user_call_sources[callee][caller_city] += 1
        self.access_frequency_matrix[callee][caller_city] += 1

        # Check local replica
        if self.enable_replication and caller_city in self.replica_locations[callee]:
            self.metrics['replica_hits'] += 1
            self.costs['search_with_replication'] += 1
            return 1, ['replica']

        # Check region-level replica
        if self.enable_replication:
            caller_region = list(self.G.predecessors(caller_city))[0]
            for rc in self.G.successors(caller_region):
                if rc in self.replica_locations[callee]:
                    self.metrics['replica_hits'] += 1
                    self.costs['search_with_replication'] += 2
                    return 2, ['replica_region']

        # Fall back to forwarding pointers
        latency, levels = self.find_user_with_forwarding(caller, callee)

        if self.enable_replication:
            self.costs['search_with_replication'] += latency
        else:
            self.costs['search_without_replication'] += latency

        return latency, levels

    # ------------------------------------------------------------------
    # MOBILITY
    # ------------------------------------------------------------------
    def move_user_realistic(self, user):
        current_city = self.user_locations[user]
        current_coords = self.city_coords[current_city]
        cities = list(self.city_coords.keys())
        distances = []
        for city in cities:
            if city != current_city:
                c = self.city_coords[city]
                d = math.sqrt((c[0] - current_coords[0]) ** 2 +
                              (c[1] - current_coords[1]) ** 2)
                distances.append(d)
            else:
                distances.append(float('inf'))
        distances = np.array(distances)
        probs = 1 / (1 + distances)
        probs = probs / probs.sum()
        return np.random.choice(cities, p=probs)

    def update_location_with_forwarding(self, user, new_city):
        old_city = self.user_locations[user]
        if old_city == new_city:
            return
        self.metrics['updates'] += 1
        self.user_mobility_frequency[user] += 1

        base = 1
        self.costs['update_without_replication'] += base
        if self.enable_replication:
            rep_cost = len(self.replica_locations[user]) * 0.5
            self.costs['update_with_replication'] += base + rep_cost
            self.costs['consistency_maintenance'] += rep_cost
        else:
            self.costs['update_with_replication'] += base

        old_region = list(self.G.predecessors(old_city))[0]
        new_region = list(self.G.predecessors(new_city))[0]

        if old_region == new_region:
            self.forwarding_pointers['city'][old_city].append({
                'user': user, 'new_location': new_city,
                'timestamp': self.metrics['queries']})
        else:
            self.forwarding_pointers['region'][old_region].append({
                'user': user, 'new_location': new_city,
                'timestamp': self.metrics['queries']})
            if len(self.forwarding_pointers['region'][old_region]) > \
                    self.max_forwarding_chain:
                self.forwarding_pointers['root']['Root'].append({
                    'user': user, 'new_location': new_city,
                    'timestamp': self.metrics['queries']})

        for level in self.forwarding_pointers:
            for node in self.forwarding_pointers[level]:
                if len(self.forwarding_pointers[level][node]) > \
                        self.max_forwarding_chain:
                    self.forwarding_pointers[level][node].pop(0)

        self.user_locations[user] = new_city
        self.G.remove_edge(old_city, user)
        self.G.add_edge(new_city, user)

    # ------------------------------------------------------------------
    # OPTIMAL LEVEL
    # ------------------------------------------------------------------
    def compute_optimal_level(self, user):
        calls = self.user_call_frequency[user]
        moves = max(self.user_mobility_frequency[user], 1)
        cmr = calls / moves
        if cmr > 10:
            return 'root'
        elif cmr > 5:
            return 'region'
        return 'city'

    # ------------------------------------------------------------------
    # ONE SIMULATION STEP
    # ------------------------------------------------------------------
    def simulate_step(self):
        step_search_cost = 0
        step_update_cost = 0

        # Phase 1 – move users
        moved_users = []
        for user in list(self.user_locations.keys()):
            if np.random.rand() < self.mobility_prob:
                new_city = self.move_user_realistic(user)
                self.update_location_with_forwarding(user, new_city)
                moved_users.append(user)
                step_update_cost += 1

        # Phase 2 – simulate calls
        users = list(self.user_locations.keys())
        calls = []
        total_latency_with = 0
        total_latency_without = 0
        num_calls = int(len(users) * self.call_prob)

        for _ in range(num_calls):
            caller = np.random.choice(users)
            callee = np.random.choice(users)
            if caller == callee:
                continue

            self.user_call_frequency[caller] += 1
            self.user_call_frequency[callee] += 1

            # Latency WITH replication (or forwarding only if disabled)
            latency_with, levels = self.find_user_with_replication(
                caller, callee)

            # Latency WITHOUT replication (forwarding only, no side effects)
            latency_without = self._forwarding_latency(caller, callee)

            total_latency_with += latency_with
            total_latency_without += latency_without
            step_search_cost += latency_with

            calls.append({
                'caller': caller,
                'callee': callee,
                'latency': latency_with,
                'latency_no_repl': latency_without,
                'forwarding_levels': levels,
                'optimal_level': self.compute_optimal_level(callee),
                'used_replica': 'replica' in levels
                                or 'replica_region' in levels
            })

        # Phase 3 – update replicas AFTER calls
        if self.enable_replication:
            for call in calls:
                self.update_replicas(call['callee'])
            for user in moved_users:
                self.update_replicas(user)

        # record costs
        self.costs['search_costs_per_step'].append(step_search_cost)
        self.costs['update_costs_per_step'].append(step_update_cost)

        # CMR
        if self.metrics['updates'] > 0:
            self.metrics['cmr_history'].append(
                self.metrics['queries'] / self.metrics['updates'])

        # latency histories
        if calls:
            self.metrics['latency_history'].append(
                total_latency_with / len(calls))
            self.metrics['latency_no_repl_history'].append(
                total_latency_without / len(calls))

        # replica count
        total_reps = sum(len(r) for r in self.replica_locations.values())
        self.metrics['replica_count_history'].append(total_reps)

        # replication benefit
        if self.enable_replication:
            s_diff = (self.costs['search_without_replication']
                      - self.costs['search_with_replication'])
            u_diff = (self.costs['update_with_replication']
                      - self.costs['update_without_replication'])
            self.metrics['replication_benefit'].append(s_diff - u_diff)

        return calls, moved_users

    # ------------------------------------------------------------------
    # FULL SIMULATION
    # ------------------------------------------------------------------
    def run_simulation(self, steps=10, seed=42):
        np.random.seed(seed)

        results = {
            'calls': [],
            'movements': [],
            'forwarding_effectiveness': defaultdict(list),
            'replication_stats': [],
            'cost_comparison': {}
        }

        for step in range(steps):
            calls, moved = self.simulate_step()
            results['calls'].extend(calls)
            results['movements'].append(len(moved))

            for level in self.metrics['forwarding_hits']:
                results['forwarding_effectiveness'][level].append(
                    self.metrics['forwarding_hits'][level])

            if self.enable_replication:
                tr = sum(len(r) for r in self.replica_locations.values())
                ur = len([u for u, r in self.replica_locations.items()
                          if len(r) > 0])
                results['replication_stats'].append({
                    'step': step,
                    'total_replicas': tr,
                    'replica_hits': self.metrics['replica_hits'],
                    'users_with_replicas': ur
                })

        results['cost_comparison'] = {
            'search_without_replication':
                self.costs['search_without_replication'],
            'search_with_replication':
                self.costs['search_with_replication'],
            'update_without_replication':
                self.costs['update_without_replication'],
            'update_with_replication':
                self.costs['update_with_replication'],
            'storage_cost': self.costs['storage_cost'],
            'consistency_cost': self.costs['consistency_maintenance']
        }
        return results

    # ------------------------------------------------------------------
    # ANALYSIS HELPERS
    # ------------------------------------------------------------------
    def get_replication_analysis(self):
        analysis = {
            'users_with_replicas': {},
            'replica_distribution': defaultdict(int),
            'access_patterns': {},
            'cmr_analysis': {}
        }
        for user in self.user_locations:
            reps = self.replica_locations[user]
            cmr = (self.user_call_frequency[user]
                   / max(self.user_mobility_frequency[user], 1))
            if reps:
                analysis['users_with_replicas'][user] = {
                    'locations': list(reps), 'count': len(reps), 'cmr': cmr}
            analysis['replica_distribution'][len(reps)] += 1

        for user, sources in self.user_call_sources.items():
            if sources:
                analysis['access_patterns'][user] = dict(sources)

        for user in self.user_locations:
            calls = self.user_call_frequency[user]
            moves = max(self.user_mobility_frequency[user], 1)
            cmr = calls / moves
            cat = ('high_cmr' if cmr > 10
                   else 'medium_cmr' if cmr > 5
                   else 'low_cmr')
            analysis['cmr_analysis'].setdefault(cat, []).append({
                'user': user, 'cmr': cmr,
                'replicas': len(self.replica_locations[user])})
        return analysis

    def get_call_data_for_map(self):
        call_data = []
        sampled = list(self.user_call_frequency.keys())[:50]
        if not sampled:
            return call_data
        for i, caller in enumerate(sampled):
            for j, callee in enumerate(sampled):
                if i != j and np.random.rand() < 0.1:
                    cc = self.user_locations[caller]
                    ce = self.user_locations[callee]
                    lat1, lon1 = self.city_coords[cc]
                    lat2, lon2 = self.city_coords[ce]
                    has_rep = cc in self.replica_locations[callee]
                    call_data.append({
                        'caller': caller, 'callee': callee,
                        'caller_city': cc, 'callee_city': ce,
                        'lat1': lat1, 'lon1': lon1,
                        'lat2': lat2, 'lon2': lon2,
                        'latency': 1 if has_rep else 5,
                        'has_replica': has_rep})
        return call_data