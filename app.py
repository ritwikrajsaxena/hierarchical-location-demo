import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from simulation import EnhancedHierarchicalSimulator
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

st.set_page_config(layout="wide", page_title="Enhanced Hierarchical Location Simulator")
st.title("🌐 Hierarchical Location Management with Replication")
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    
    st.subheader("Network Topology")
    num_regions = st.slider("Number of Regions", 2, 8, 3)
    cities_per_region = st.slider("Cities per Region", 2, 8, 4)
    users_per_city = st.slider("Users per City", 1, 15, 8)
    
    st.subheader("Behavior Settings")
    mobility_prob = st.slider("Mobility Probability", 0.0, 0.5, 0.15, 0.05)
    call_prob = st.slider("Call Probability", 0.0, 0.5, 0.3, 0.05)
    
    st.subheader("Forwarding Configuration")
    max_forwarding_chain = st.slider("Max Forwarding Chain Length", 1, 10, 5)
    
    st.subheader("🔄 Replication Settings")
    enable_replication = st.checkbox("Enable Replication", value=True)
    if enable_replication:
        replication_strategy = st.selectbox(
            "Replication Strategy",
            ["CMR-based", "Access-frequency", "Hybrid"]
        )
        replication_threshold = st.slider("Replication Threshold", 1, 20, 5)
        max_replicas = st.slider("Max Replicas per User", 1, 5, 3)
    else:
        replication_strategy = "CMR-based"
        replication_threshold = 5
        max_replicas = 3
    
    st.subheader("Simulation Control")
    simulation_steps = st.slider("Simulation Steps", 1, 20, 10)
    
    st.markdown("---")
    run_simulation = st.button("🚀 Run Simulation", use_container_width=True)
    
    # Display network size
    total_users = num_regions * cities_per_region * users_per_city
    st.info(f"📊 Total Network Size: {total_users} users")

# --- Initialize Session State ---
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
    st.session_state.simulator = None
    st.session_state.sim_no_repl = None
    st.session_state.results_no_repl = None
    st.session_state.last_calls = []

# --- Run Simulation ---
if run_simulation:
    with st.spinner('Running hierarchical location simulation...'):
        # Run simulation WITH replication
        sim_with = EnhancedHierarchicalSimulator(
            num_regions=num_regions,
            cities_per_region=cities_per_region,
            users_per_city=users_per_city,
            mobility_prob=mobility_prob,
            call_prob=call_prob,
            max_forwarding_chain=max_forwarding_chain,
            enable_replication=enable_replication,
            replication_threshold=replication_threshold,
            max_replicas=max_replicas,
            replication_strategy=replication_strategy
        )
        
        # Track calls for visualization
        all_calls = []
        for step in range(simulation_steps):
            calls, moved_users = sim_with.simulate_step()
            # Store calls with location information
            for call in calls:
                caller_city = sim_with.user_locations[call['caller']]
                callee_city = sim_with.user_locations[call['callee']]
                call['caller_city'] = caller_city
                call['callee_city'] = callee_city
                call['caller_coords'] = sim_with.city_coords[caller_city]
                call['callee_coords'] = sim_with.city_coords[callee_city]
                call['step'] = step
                all_calls.append(call)
        
        results_with = {
            'calls': all_calls,
            'movements': [],
            'forwarding_effectiveness': sim_with.metrics['forwarding_hits'],
            'replication_stats': [],
            'cost_comparison': {
                'search_without_replication': sim_with.costs['search_without_replication'],
                'search_with_replication': sim_with.costs['search_with_replication'],
                'update_without_replication': sim_with.costs['update_without_replication'],
                'update_with_replication': sim_with.costs['update_with_replication'],
                'storage_cost': sim_with.costs['storage_cost'],
                'consistency_cost': sim_with.costs['consistency_maintenance']
            }
        }
        
        # Run simulation WITHOUT replication for comparison
        sim_without = EnhancedHierarchicalSimulator(
            num_regions=num_regions,
            cities_per_region=cities_per_region,
            users_per_city=users_per_city,
            mobility_prob=mobility_prob,
            call_prob=call_prob,
            max_forwarding_chain=max_forwarding_chain,
            enable_replication=False
        )
        results_without = sim_without.run_simulation(steps=simulation_steps)
        
        # Store in session state
        st.session_state.sim_results = results_with
        st.session_state.simulator = sim_with
        st.session_state.sim_no_repl = sim_without
        st.session_state.results_no_repl = results_without
        st.session_state.last_calls = all_calls[-min(50, len(all_calls)):] if all_calls else []  # Store last 50 calls
        
        st.success("✅ Simulation completed successfully!")

# --- Display Results ---
if st.session_state.sim_results is not None:
    sim = st.session_state.simulator
    results = st.session_state.sim_results
    sim_no_repl = st.session_state.sim_no_repl
    results_no_repl = st.session_state.results_no_repl
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📍 Geographic View", 
        "📊 Performance Metrics", 
        "🌳 Network Hierarchy",
        "📈 Forwarding Analysis",
        "📋 Detailed Statistics",
        "🔄 Replication Strategy",
        "💰 Cost Comparison",
        "⚖️ Trade-off Analysis"
    ])
    
    with tab1:
        st.header("Geographic Distribution & Call Patterns")
        
        # Prepare city data
        city_data = []
        for city, coords in sim.city_coords.items():
            user_count = sum(1 for u in sim.user_locations.values() if u == city)
            replica_count = sum(1 for u, replicas in sim.replica_locations.items() if city in replicas)
            city_data.append({
                'city': city,
                'lat': coords[0],
                'lon': coords[1],
                'users': user_count,
                'replicas': replica_count,
                'region': city.split('_')[1]
            })
        
        city_df = pd.DataFrame(city_data)
        
        # Main map with users and call connections
        st.subheader("🗺️ Live Call Activity Map")
        
        # Create the main figure
        fig = go.Figure()
        
        # Add city markers
        fig.add_trace(go.Scattermapbox(
            lat=city_df['lat'],
            lon=city_df['lon'],
            mode='markers',
            marker=dict(
                size=city_df['users'] * 3 + 10,
                color=city_df['users'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Users", x=1.02),
                opacity=0.8
            ),
            text=city_df['city'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Users: %{marker.color}<br>' +
                         'Lat: %{lat}<br>' +
                         'Lon: %{lon}<extra></extra>',
            name='Cities'
        ))
        
        # Add call lines
        if st.session_state.last_calls:
            # Create lines for each call
            for call in st.session_state.last_calls[-20:]:  # Show last 20 calls
                caller_coords = call['caller_coords']
                callee_coords = call['callee_coords']
                
                # Determine line color based on call characteristics
                if call.get('used_replica', False):
                    line_color = 'green'
                    line_width = 2
                    line_opacity = 0.6
                elif call['latency'] > 3:
                    line_color = 'red'
                    line_width = 1.5
                    line_opacity = 0.4
                else:
                    line_color = 'blue'
                    line_width = 1
                    line_opacity = 0.3
                
                # Add line trace
                fig.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=[caller_coords[1], callee_coords[1]],
                    lat=[caller_coords[0], callee_coords[0]],
                    line=dict(
                        width=line_width,
                        color=line_color
                    ),
                    opacity=line_opacity,
                    hovertemplate=f'Call from {call["caller"]} to {call["callee"]}<br>' +
                                 f'Latency: {call["latency"]} hops<br>' +
                                 f'Replica Used: {call.get("used_replica", False)}<extra></extra>',
                    showlegend=False
                ))
        
        # Update map layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center=dict(
                    lat=city_df['lat'].mean(),
                    lon=city_df['lon'].mean()
                ),
                zoom=3
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=600,
            title="Active Call Connections (Last 20 Calls)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Call statistics by city
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📞 Total Calls", len(results['calls']))
        
        with col2:
            if st.session_state.last_calls:
                avg_latency = np.mean([c['latency'] for c in st.session_state.last_calls])
                st.metric("⏱️ Avg Latency (recent)", f"{avg_latency:.2f} hops")
        
        with col3:
            if enable_replication and st.session_state.last_calls:
                replica_calls = sum(1 for c in st.session_state.last_calls if c.get('used_replica', False))
                st.metric("✅ Replica Hits (recent)", replica_calls)
        
        # Call heatmap
        st.subheader("📊 Call Intensity Heatmap")
        
        # Create call matrix between cities
        call_matrix = pd.DataFrame(0, index=list(sim.city_coords.keys()), columns=list(sim.city_coords.keys()))
        
        for call in results['calls']:
            if 'caller_city' in call and 'callee_city' in call:
                call_matrix.loc[call['caller_city'], call['callee_city']] += 1
        
        # Create heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=call_matrix.values,
            x=call_matrix.columns,
            y=call_matrix.index,
            colorscale='YlOrRd',
            hovertemplate='From: %{y}<br>To: %{x}<br>Calls: %{z}<extra></extra>'
        ))
        
        fig_heat.update_layout(
            title="Call Frequency Between Cities",
            xaxis_title="Destination City",
            yaxis_title="Origin City",
            height=500
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with tab2:
        st.header("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", sim.metrics['queries'])
        with col2:
            st.metric("Total Updates", sim.metrics['updates'])
        with col3:
            if sim.metrics['updates'] > 0:
                cmr = sim.metrics['queries'] / sim.metrics['updates']
                st.metric("CMR", f"{cmr:.2f}")
        with col4:
            if enable_replication:
                st.metric("Replica Hits", sim.metrics['replica_hits'])
            else:
                st.metric("Total Calls", len(results['calls']))
        
        # Latency comparison
        if sim.metrics['latency_history'] and sim_no_repl.metrics['latency_history']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sim.metrics['latency_history'],
                mode='lines+markers',
                name='With Replication' if enable_replication else 'Current',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=sim_no_repl.metrics['latency_history'],
                mode='lines+markers',
                name='Without Replication',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(
                title="Average Call Latency Comparison",
                xaxis_title="Time Step",
                yaxis_title="Average Latency (hops)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Network Hierarchy Visualization")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌍 Regions: {num_regions}")
        with col2:
            st.info(f"🏙️ Total Cities: {num_regions * cities_per_region}")
        with col3:
            st.info(f"👥 Total Users: {num_regions * cities_per_region * users_per_city}")
        
        # Tree structure bar chart and actual tree visualization
        col1, col2 = st.columns(2)
        
        with col1:
            tree_data = {
                'Level': ['Root', 'Region', 'City', 'User'],
                'Count': [
                    1,
                    num_regions,
                    num_regions * cities_per_region,
                    num_regions * cities_per_region * users_per_city
                ]
            }
            
            tree_df = pd.DataFrame(tree_data)
            
            fig = px.bar(
                tree_df,
                x='Level',
                y='Count',
                title='Node Distribution by Hierarchy Level',
                color='Level',
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌳 Hierarchical Tree Structure")
            
            # Create a smaller sample tree for visualization
            sample_tree = nx.DiGraph()
            
            # Add root
            sample_tree.add_node("Root", level=0)
            
            # Add sample regions (show all if small, sample if large)
            regions_to_show = min(num_regions, 3)
            for r in range(regions_to_show):
                region = f"R{r}"
                sample_tree.add_node(region, level=1)
                sample_tree.add_edge("Root", region)
                
                # Add sample cities
                cities_to_show = min(cities_per_region, 3)
                for c in range(cities_to_show):
                    city = f"C{r}_{c}"
                    sample_tree.add_node(city, level=2)
                    sample_tree.add_edge(region, city)
                    
                    # Add sample users
                    users_to_show = min(users_per_city, 2)
                    for u in range(users_to_show):
                        user = f"U{r}_{c}_{u}"
                        sample_tree.add_node(user, level=3)
                        sample_tree.add_edge(city, user)
            
            # Calculate positions for tree layout
            pos = nx.spring_layout(sample_tree, k=2, iterations=50, seed=42)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in sample_tree.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            
            for node in sample_tree.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Color by level
                level = sample_tree.nodes[node]['level']
                colors = ['red', 'orange', 'green', 'blue']
                node_color.append(colors[level])
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    color=node_color,
                    size=15,
                    line_width=2
                )
            )
            
            # Create figure
            fig_tree = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title=f'Tree Structure (showing {regions_to_show} regions)',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500
                        ))
            
            st.plotly_chart(fig_tree, use_container_width=True)
            
            if num_regions > 3:
                st.info(f"Showing sample of {regions_to_show} regions out of {num_regions} total")
        
        # Add hierarchical path visualization
        st.subheader("📍 Sample Location Paths")
        
        # Show some example paths from users to root
        sample_users = list(sim.user_locations.keys())[:3]  # Show 3 sample users
        
        path_data = []
        for user in sample_users:
            city = sim.user_locations[user]
            region = list(sim.G.predecessors(city))[0]
            
            path_data.append({
                'User': user,
                'Path': f"{user} → {city} → {region} → Root",
                'Current City': city,
                'Hops to Root': 3
            })
        
        if path_data:
            path_df = pd.DataFrame(path_data)
            st.table(path_df)
    
    with tab4:
        st.header("Forwarding Pointer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forwarding Pointer Usage")
            
            if sim.metrics['forwarding_hits']:
                fp_data = pd.DataFrame({
                    'Level': list(sim.metrics['forwarding_hits'].keys()),
                    'Hits': list(sim.metrics['forwarding_hits'].values())
                })
                
                fig = px.bar(
                    fp_data,
                    x='Level',
                    y='Hits',
                    title='Forwarding Pointer Hits by Level',
                    color='Level',
                    color_discrete_map={'city': 'blue', 'region': 'green', 'root': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Latency Distribution")
            
            if results['calls']:
                latencies = [call['latency'] for call in results['calls']]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=latencies,
                    nbinsx=20,
                    name='Latency Distribution',
                    marker_color='teal'
                ))
                fig.update_layout(
                    title="Call Latency Distribution",
                    xaxis_title="Latency (hops)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Detailed Statistics")
        
        if results['calls']:
            call_stats = pd.DataFrame(results['calls'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_latency = call_stats['latency'].mean()
                st.metric("Avg Latency", f"{avg_latency:.2f} hops")
            
            with col2:
                max_latency = call_stats['latency'].max()
                st.metric("Max Latency", f"{max_latency} hops")
            
            with col3:
                if enable_replication and 'used_replica' in call_stats.columns:
                    replica_usage = call_stats['used_replica'].sum()
                    st.metric("Calls Using Replicas", replica_usage)
                else:
                    min_latency = call_stats['latency'].min()
                    st.metric("Min Latency", f"{min_latency} hops")
    
    with tab6:
        st.header("🔄 Replication Strategy Analysis")
        
        if enable_replication:
            repl_analysis = sim.get_replication_analysis()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Replica Distribution")
                
                # Users with replicas
                users_with_replicas = len(repl_analysis['users_with_replicas'])
                total_replicas = sum(len(sim.replica_locations[u]) for u in sim.replica_locations)
                
                st.metric("Users with Replicas", users_with_replicas)
                st.metric("Total Replicas", total_replicas)
                
                # Replica count distribution
                if repl_analysis['replica_distribution']:
                    dist_df = pd.DataFrame(
                        list(repl_analysis['replica_distribution'].items()),
                        columns=['Replica Count', 'Users']
                    )
                    
                    fig = px.bar(
                        dist_df,
                        x='Replica Count',
                        y='Users',
                        title='Distribution of Replica Counts'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("CMR-based Analysis")
                
                # CMR categories
                cmr_stats = {}
                for category, users in repl_analysis['cmr_analysis'].items():
                    if users:
                        avg_replicas = np.mean([u['replicas'] for u in users])
                        cmr_stats[category] = {
                            'count': len(users),
                            'avg_replicas': avg_replicas
                        }
                
                if cmr_stats:
                    cmr_df = pd.DataFrame.from_dict(cmr_stats, orient='index')
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Users by CMR Category', 'Avg Replicas by CMR')
                    )
                    
                    fig.add_trace(
                        go.Bar(x=cmr_df.index, y=cmr_df['count'], name='User Count'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=cmr_df.index, y=cmr_df['avg_replicas'], name='Avg Replicas'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(title='CMR-based Replication Analysis')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Replica evolution over time
            if sim.metrics['replica_count_history']:
                st.subheader("Replica Evolution")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=sim.metrics['replica_count_history'],
                    mode='lines+markers',
                    name='Total Replicas',
                    line=dict(color='purple', width=2)
                ))
                fig.update_layout(
                    title="Total Replicas Over Time",
                    xaxis_title="Time Step",
                    yaxis_title="Number of Replicas"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable replication to see replication strategy analysis")
    
    with tab7:
        st.header("💰 Cost Comparison Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Search vs Update Costs")
            
            # Compare costs with and without replication
            cost_data = {
                'Cost Type': ['Search', 'Update', 'Storage', 'Consistency'],
                'Without Replication': [
                    sim_no_repl.costs['search_without_replication'],
                    sim_no_repl.costs['update_without_replication'],
                    0,
                    0
                ],
                'With Replication': [
                    sim.costs['search_with_replication'] if enable_replication else sim.costs['search_without_replication'],
                    sim.costs['update_with_replication'] if enable_replication else sim.costs['update_without_replication'],
                    sim.costs['storage_cost'] if enable_replication else 0,
                    sim.costs['consistency_maintenance'] if enable_replication else 0
                ]
            }
            
            cost_df = pd.DataFrame(cost_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cost_df['Cost Type'],
                y=cost_df['Without Replication'],
                name='Without Replication',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=cost_df['Cost Type'],
                y=cost_df['With Replication'],
                name='With Replication',
                marker_color='darkblue'
            ))
            fig.update_layout(
                title='Cost Comparison',
                yaxis_title='Cost Units',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost Evolution")
            
            if sim.costs['search_costs_per_step'] and sim.costs['update_costs_per_step']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=sim.costs['search_costs_per_step'],
                    mode='lines',
                    name='Search Cost',
                    line=dict(color='green', width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=sim.costs['update_costs_per_step'],
                    mode='lines',
                    name='Update Cost',
                    line=dict(color='orange', width=2)
                ))
                fig.update_layout(
                    title='Cost Evolution Over Time',
                    xaxis_title='Time Step',
                    yaxis_title='Cost per Step'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Cost savings analysis
        if enable_replication:
            st.subheader("Cost Savings Analysis")
            
            search_savings = sim_no_repl.costs['search_without_replication'] - sim.costs['search_with_replication']
            update_overhead = sim.costs['update_with_replication'] - sim.costs['update_without_replication']
            net_benefit = search_savings - update_overhead - sim.costs['storage_cost']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Search Cost Savings", f"{search_savings:.0f}")
            with col2:
                st.metric("Update Overhead", f"{update_overhead:.0f}")
            with col3:
                color = "normal" if net_benefit > 0 else "inverse"
                st.metric("Net Benefit", f"{net_benefit:.0f}", delta_color=color)
    
    with tab8:
        st.header("⚖️ Trade-off Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Search vs Update Trade-off")
            
            # Create trade-off scatter plot
            strategies = []
            
            # No replication point
            strategies.append({
                'Strategy': 'No Replication',
                'Search Cost': sim_no_repl.costs['search_without_replication'],
                'Update Cost': sim_no_repl.costs['update_without_replication']
            })
            
            # With replication point
            if enable_replication:
                strategies.append({
                    'Strategy': f'Replication (threshold={replication_threshold})',
                    'Search Cost': sim.costs['search_with_replication'],
                    'Update Cost': sim.costs['update_with_replication']
                })
            
            strategy_df = pd.DataFrame(strategies)
            
            fig = px.scatter(
                strategy_df,
                x='Update Cost',
                y='Search Cost',
                text='Strategy',
                title='Search vs Update Cost Trade-off',
                size_max=20
            )
            fig.update_traces(textposition='top center', marker=dict(size=15))
            fig.update_layout(
                xaxis_title='Update Cost',
                yaxis_title='Search Cost',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Efficiency Metrics")
            
            # Calculate efficiency metrics
            if enable_replication:
                total_ops_no_repl = sim_no_repl.costs['search_without_replication'] + \
                                   sim_no_repl.costs['update_without_replication']
                total_ops_with_repl = sim.costs['search_with_replication'] + \
                                     sim.costs['update_with_replication'] + \
                                     sim.costs['storage_cost']
                
                efficiency_gain = ((total_ops_no_repl - total_ops_with_repl) / total_ops_no_repl) * 100
                
                metrics_data = {
                    'Metric': ['Total Operations (No Repl)', 'Total Operations (With Repl)', 
                              'Efficiency Gain', 'Replica Hit Rate'],
                    'Value': [
                        f"{total_ops_no_repl:.0f}",
                        f"{total_ops_with_repl:.0f}",
                        f"{efficiency_gain:.1f}%",
                        f"{(sim.metrics['replica_hits'] / max(len(results['calls']), 1) * 100):.1f}%"
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics_data)
                st.table(metrics_df)
        
        # Summary recommendations
        st.subheader("📌 Recommendations")
        
        if enable_replication:
            search_savings = sim_no_repl.costs['search_without_replication'] - sim.costs['search_with_replication']
            update_overhead = sim.costs['update_with_replication'] - sim.costs['update_without_replication']
            net_benefit = search_savings - update_overhead - sim.costs['storage_cost']
            
            if net_benefit > 0:
                st.success(f"""
                ✅ **Replication is beneficial** for this configuration:
                - Net benefit: {net_benefit:.0f} cost units saved
                - Search cost reduced by {search_savings:.0f}
                - Replica hit rate: {(sim.metrics['replica_hits'] / max(len(results['calls']), 1) * 100):.1f}%
                """)
            else:
                st.warning(f"""
                ⚠️ **Replication may not be optimal** for this configuration:
                - Net cost: {abs(net_benefit):.0f} additional cost units
                - Consider adjusting the replication threshold
                - Current threshold: {replication_threshold}
                """)

else:
    # Welcome screen
    st.info("👈 Configure simulation parameters in the sidebar and click 'Run Simulation' to start!")
    
    st.markdown("""
    ### 📖 About This System
    
    This enhanced hierarchical location management system now includes:
    
    - **Interactive Map Visualization**: See real-time call connections on a geographic map
    - **Hierarchical Tree Visualization**: View the network structure as a tree
    - **Selective Replication**: Intelligently replicates user data based on access patterns
    - **Call Pattern Analysis**: Visualize communication patterns between cities
    
    ### 🎯 Key Features
    
    1. **Live Call Activity Map**: Shows active calls as lines connecting cities
    2. **Network Hierarchy Tree**: Visualizes the Root → Region → City → User structure
    3. **Color-coded Connections**: 
        - 🟢 Green lines: Calls using replicas (fast)
        - 🔵 Blue lines: Normal latency calls
        - 🔴 Red lines: High latency calls
    4. **Call Intensity Heatmap**: Shows frequency of calls between cities
    
    ### 🚀 Getting Started
    
    1. Configure network topology in the sidebar
    2. Set mobility and call patterns
    3. Enable replication and choose strategy
    4. Run simulation to see the interactive visualizations!
    """)