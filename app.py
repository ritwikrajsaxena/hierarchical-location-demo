import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from simulation import EnhancedHierarchicalSimulator
import matplotlib.pyplot as plt
import seaborn as sns

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
        results_with = sim_with.run_simulation(steps=simulation_steps)
        
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
        
        # Map 1: User Distribution
        st.subheader("🗺️ User Distribution Map")
        
        # Prepare city data
        city_data = []
        for city, coords in sim.city_coords.items():
            user_count = sum(1 for u in sim.user_locations.values() if u == city)
            region_id = int(city.split('_')[1])
            city_data.append({
                'city': city,
                'lat': coords[0],
                'lon': coords[1],
                'users': user_count,
                'region': f"Region_{region_id}"
            })
        
        city_df = pd.DataFrame(city_data)
        
        if not city_df.empty:
            fig1 = px.scatter_mapbox(
                city_df,
                lat="lat",
                lon="lon",
                hover_name="city",
                hover_data={'users': True, 'region': True, 'lat': ':.2f', 'lon': ':.2f'},
                color="users",
                size="users",
                size_max=30,
                zoom=3.5,
                center={"lat": 39.0, "lon": -98.0},
                mapbox_style="open-street-map",
                title="User Distribution Across Cities",
                color_continuous_scale="Viridis",
                labels={'users': 'User Count'},
                height=600
            )
            fig1.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Statistics for User Distribution
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📍 Total Cities", len(city_df))
            with col2:
                st.metric("👥 Total Users", city_df['users'].sum())
            with col3:
                st.metric("👥 Avg Users/City", f"{city_df['users'].mean():.1f}")
        
        st.markdown("---")
        
        # Map 2: Replica Distribution
        st.subheader("🔄 Replica Distribution Map")
        
        if enable_replication:
            # Prepare replica data
            replica_data = []
            for city, coords in sim.city_coords.items():
                user_count = sum(1 for u in sim.user_locations.values() if u == city)
                replica_count = sum(1 for u, replicas in sim.replica_locations.items() if city in replicas)
                replica_data.append({
                    'city': city,
                    'lat': coords[0],
                    'lon': coords[1],
                    'replicas': replica_count,
                    'users': user_count,
                    'has_replicas': 'Yes' if replica_count > 0 else 'No'
                })
            
            replica_df = pd.DataFrame(replica_data)
            
            if not replica_df.empty:
                fig2 = px.scatter_mapbox(
                    replica_df,
                    lat="lat",
                    lon="lon",
                    hover_name="city",
                    hover_data={'replicas': True, 'users': True, 'lat': ':.2f', 'lon': ':.2f'},
                    color="replicas",
                    size="replicas",
                    size_max=35,
                    zoom=3.5,
                    center={"lat": 39.0, "lon": -98.0},
                    mapbox_style="open-street-map",
                    title="Replica Distribution Across Cities",
                    color_continuous_scale="Reds",
                    labels={'replicas': 'Replica Count'},
                    height=600
                )
                # Ensure cities without replicas are still visible
                fig2.update_traces(marker=dict(sizemin=5))
                fig2.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig2, use_container_width=True)
                
                # Statistics for Replica Distribution
                col1, col2, col3 = st.columns(3)
                total_replicas = replica_df['replicas'].sum()
                cities_with_replicas = len(replica_df[replica_df['replicas'] > 0])
                with col1:
                    st.metric("💾 Total Replicas", total_replicas)
                with col2:
                    st.metric("🏙️ Cities with Replicas", cities_with_replicas)
                with col3:
                    if cities_with_replicas > 0:
                        st.metric("💾 Avg Replicas/City", f"{total_replicas/cities_with_replicas:.1f}")
        else:
            st.info("Enable replication to see replica distribution")
        
        st.markdown("---")
        
        # Map 3: Call Patterns
        st.subheader("📞 Call Patterns Map")
        
        # Get call data for visualization
        call_data = sim.get_call_data_for_map()
        
        if call_data:
            call_df = pd.DataFrame(call_data)
            
            # Create map with call lines
            fig3 = go.Figure()
            
            # Add city markers first
            for _, city in city_df.iterrows():
                fig3.add_trace(go.Scattermapbox(
                    lon=[city['lon']],
                    lat=[city['lat']],
                    mode='markers',
                    marker=dict(size=10, color='lightblue'),
                    name=city['city'],
                    hovertext=f"{city['city']}: {city['users']} users",
                    showlegend=False
                ))
            
            # Count calls by latency category
            low_latency_calls = 0
            med_latency_calls = 0
            high_latency_calls = 0
            
            # Add call lines with color based on latency
            for _, call in call_df.iterrows():
                if call['latency'] <= 2:
                    color = 'green'
                    width = 1.5
                    low_latency_calls += 1
                elif call['latency'] <= 4:
                    color = 'orange'
                    width = 2
                    med_latency_calls += 1
                else:
                    color = 'red'
                    width = 2.5
                    high_latency_calls += 1
                
                fig3.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=[call['lon1'], call['lon2']],
                    lat=[call['lat1'], call['lat2']],
                    line=dict(width=width, color=color),
                    hovertext=f"Latency: {call['latency']} hops",
                    showlegend=False,
                    opacity=0.7
                ))
            
            # Add legend traces
            fig3.add_trace(go.Scattermapbox(
                lon=[None], lat=[None],
                mode='markers',
                marker=dict(size=10, color='green'),
                name=f'Low Latency (1-2 hops): {low_latency_calls} calls'
            ))
            fig3.add_trace(go.Scattermapbox(
                lon=[None], lat=[None],
                mode='markers',
                marker=dict(size=10, color='orange'),
                name=f'Medium Latency (3-4 hops): {med_latency_calls} calls'
            ))
            fig3.add_trace(go.Scattermapbox(
                lon=[None], lat=[None],
                mode='markers',
                marker=dict(size=10, color='red'),
                name=f'High Latency (5+ hops): {high_latency_calls} calls'
            ))
            
            fig3.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    zoom=3.5,
                    center=dict(lat=39.0, lon=-98.0)
                ),
                title="Call Patterns and Latency Visualization",
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0},
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Statistics for Call Patterns
            col1, col2, col3 = st.columns(3)
            if len(call_df) > 0:
                avg_latency = call_df['latency'].mean()
                with col1:
                    st.metric("📊 Sample Calls Shown", len(call_df))
                with col2:
                    st.metric("⚡ Average Latency", f"{avg_latency:.1f} hops")
                with col3:
                    st.metric("📞 Total Calls Made", len(results['calls']))
        else:
            st.info("No calls to display yet. Run the simulation to generate call data.")
        
        st.markdown("---")
        
        # Map 4: Combined Network View
        st.subheader("📊 Combined Network Overview")
        
        # Create a comprehensive map showing all elements
        fig_combined = go.Figure()
        
        # Base layer: Cities with size based on users
        for _, city in city_df.iterrows():
            size = 15 + (city['users'] / city_df['users'].max()) * 20
            fig_combined.add_trace(go.Scattermapbox(
                lon=[city['lon']],
                lat=[city['lat']],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color='blue',
                    opacity=0.7,
                    line=dict(width=1, color='darkblue')
                ),
                text=f"{city['city'].split('_')[2]}",
                textfont=dict(size=10, color='white'),
                textposition="middle center",
                name='Cities',
                hovertext=f"{city['city']}<br>Users: {city['users']}<br>Region: {city['region']}",
                showlegend=False
            ))
        
        # Overlay: Replicas
        if enable_replication:
            replica_cities = []
            for city, coords in sim.city_coords.items():
                replica_count = sum(1 for u, replicas in sim.replica_locations.items() if city in replicas)
                if replica_count > 0:
                    replica_cities.append({
                        'city': city,
                        'lon': coords[1],
                        'lat': coords[0],
                        'count': replica_count
                    })
            
            if replica_cities:
                replica_overlay_df = pd.DataFrame(replica_cities)
                for _, replica in replica_overlay_df.iterrows():
                    size = 20 + (replica['count'] / replica_overlay_df['count'].max()) * 25
                    fig_combined.add_trace(go.Scattermapbox(
                        lon=[replica['lon']],
                        lat=[replica['lat']],
                        mode='markers',
                        marker=dict(
                            size=size,
                            color='red',
                            opacity=0.4,
                            line=dict(width=2, color='darkred')
                        ),
                        name='Replicas',
                        hovertext=f"{replica['city']}<br>Replicas: {replica['count']}",
                        showlegend=False
                    ))
        
        # Add sample call connections (lighter to not overcrowd)
        if call_data and len(call_data) > 0:
            sample_calls = call_df.sample(n=min(20, len(call_df)))
            for _, call in sample_calls.iterrows():
                color = 'rgba(0,255,0,0.3)' if call['latency'] <= 2 else 'rgba(255,165,0,0.3)' if call['latency'] <= 4 else 'rgba(255,0,0,0.3)'
                fig_combined.add_trace(go.Scattermapbox(
                    mode='lines',
                    lon=[call['lon1'], call['lon2']],
                    lat=[call['lat1'], call['lat2']],
                    line=dict(width=1, color=color),
                    showlegend=False,
                    hovertext=f"Call latency: {call['latency']} hops"
                ))
        
        # Add legend items
        fig_combined.add_trace(go.Scattermapbox(
            lon=[None], lat=[None],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Cities (size = users)'
        ))
        
        if enable_replication:
            fig_combined.add_trace(go.Scattermapbox(
                lon=[None], lat=[None],
                mode='markers',
                marker=dict(size=15, color='red', opacity=0.4),
                name='Replica Locations'
            ))
        
        fig_combined.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=3.5,
                center=dict(lat=39.0, lon=-98.0),
                bounds=dict(
                    west=-125,
                    east=-65,
                    south=25,
                    north=50
                )
            ),
            title="Complete Hierarchical Network Visualization",
            height=700,
            margin={"r":0,"t":50,"l":0,"b":0},
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Summary statistics for the combined view
        st.subheader("📊 Network Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"🌐 Total Network Nodes: {1 + num_regions + len(city_df) + total_users}")
        with col2:
            st.info(f"📡 Active Connections: {len(results['calls'])}")
        with col3:
            if enable_replication:
                st.info(f"💾 Replication Coverage: {(cities_with_replicas/len(city_df)*100):.1f}%")
            else:
                st.info(f"🔄 Forwarding Enabled: {max_forwarding_chain} levels")
        with col4:
            if sim.metrics['updates'] > 0:
                st.info(f"📈 System CMR: {sim.metrics['queries']/sim.metrics['updates']:.2f}")
    
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
        
        # Tree structure
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
        
        # Replication benefit over time
        if enable_replication and sim.metrics['replication_benefit']:
            st.subheader("Replication Benefit Evolution")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sim.metrics['replication_benefit'],
                mode='lines+markers',
                name='Cumulative Benefit',
                line=dict(color='green' if sim.metrics['replication_benefit'][-1] > 0 else 'red', width=2)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Cumulative Replication Benefit Over Time",
                xaxis_title="Time Step",
                yaxis_title="Benefit (positive = good)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Export options
        st.subheader("📊 Export Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Cost Analysis"):
                export_data = pd.DataFrame({
                    'Metric': ['Search Cost (No Repl)', 'Update Cost (No Repl)',
                              'Search Cost (With Repl)', 'Update Cost (With Repl)',
                              'Storage Cost', 'Consistency Cost', 'Net Benefit'],
                    'Value': [
                        sim_no_repl.costs['search_without_replication'],
                        sim_no_repl.costs['update_without_replication'],
                        sim.costs['search_with_replication'] if enable_replication else 0,
                        sim.costs['update_with_replication'] if enable_replication else 0,
                        sim.costs['storage_cost'] if enable_replication else 0,
                        sim.costs['consistency_maintenance'] if enable_replication else 0,
                        net_benefit if enable_replication else 0
                    ]
                })
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="Download Cost Analysis CSV",
                    data=csv,
                    file_name='replication_cost_analysis.csv',
                    mime='text/csv'
                )
        
        with col2:
            if enable_replication and st.button("Export Replication Stats"):
                repl_stats_df = pd.DataFrame(results['replication_stats'])
                csv = repl_stats_df.to_csv(index=False)
                st.download_button(
                    label="Download Replication Stats CSV",
                    data=csv,
                    file_name='replication_statistics.csv',
                    mime='text/csv'
                )

else:
    # Welcome screen
    st.info("👈 Configure simulation parameters in the sidebar and click 'Run Simulation' to start!")
    
    st.markdown("""
    ### 📖 About This System
    
    This enhanced hierarchical location management system now includes:
    
    - **Selective Replication**: Intelligently replicates user data based on access patterns
    - **Cost Analysis**: Comprehensive comparison of search vs update costs
    - **Trade-off Visualization**: Clear visualization of the replication trade-offs
    - **Multiple Strategies**: CMR-based, access-frequency, and hybrid replication strategies
    
    ### 🎯 New Features
    
    1. **Replication Management**: Dynamic replica placement based on user behavior
    2. **Cost Tracking**: Detailed cost analysis for search, update, storage, and consistency
    3. **Benefit Analysis**: Real-time calculation of replication benefits
    4. **Strategy Comparison**: Side-by-side comparison with and without replication
    
    ### 📊 Key Metrics Tracked
    
    - **Search Cost**: Cost of finding users (reduced by replicas)
    - **Update Cost**: Cost of updating locations (increased by replicas)
    - **Storage Cost**: Cost of maintaining replicas
    - **Consistency Cost**: Cost of keeping replicas synchronized
    - **Net Benefit**: Overall benefit of replication strategy
    
    ### 🚀 Getting Started
    
    1. Configure network topology
    2. Set mobility and call patterns
    3. Enable replication and choose strategy
    4. Run simulation to see comprehensive analysis!
    """)