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
st.title("🌐 Enhanced Hierarchical Location Management System")
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
    
    st.subheader("Simulation Control")
    simulation_steps = st.slider("Simulation Steps", 1, 20, 5)
    
    st.markdown("---")
    run_simulation = st.button("🚀 Run Simulation", use_container_width=True)
    
    # Display network size
    total_users = num_regions * cities_per_region * users_per_city
    st.info(f"📊 Total Network Size: {total_users} users")

# --- Initialize Session State ---
if 'sim_results' not in st.session_state:
    st.session_state.sim_results = None
    st.session_state.simulator = None

# --- Run Simulation ---
if run_simulation:
    with st.spinner('Running hierarchical location simulation...'):
        # Create simulator
        sim = EnhancedHierarchicalSimulator(
            num_regions=num_regions,
            cities_per_region=cities_per_region,
            users_per_city=users_per_city,
            mobility_prob=mobility_prob,
            call_prob=call_prob,
            max_forwarding_chain=max_forwarding_chain
        )
        
        # Run simulation
        results = sim.run_simulation(steps=simulation_steps)
        
        # Store in session state
        st.session_state.sim_results = results
        st.session_state.simulator = sim
        
        st.success("✅ Simulation completed successfully!")

# --- Display Results ---
if st.session_state.sim_results is not None:
    sim = st.session_state.simulator
    results = st.session_state.sim_results
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📍 Geographic View", 
        "📊 Performance Metrics", 
        "🌳 Network Hierarchy",
        "📈 Forwarding Analysis",
        "📋 Detailed Statistics"
    ])
    
    with tab1:
        st.header("Geographic Distribution & Call Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User Distribution Map")
            
            # Create city data for map
            city_data = []
            for city, coords in sim.city_coords.items():
                user_count = sum(1 for u in sim.user_locations.values() if u == city)
                city_data.append({
                    'city': city,
                    'lat': coords[0],
                    'lon': coords[1],
                    'users': user_count,
                    'region': city.split('_')[1]
                })
            
            city_df = pd.DataFrame(city_data)
            
            if not city_df.empty:
                fig = px.scatter_mapbox(
                    city_df,
                    lat="lat",
                    lon="lon",
                    hover_name="city",
                    hover_data=["users"],
                    color="region",
                    size="users",
                    size_max=20,
                    zoom=3,
                    mapbox_style="carto-positron",
                    title="User Distribution Across Cities"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Call Pattern Visualization")
            
            # Extract call data
            if results['calls']:
                call_data = []
                for call in results['calls'][:50]:  # Limit to 50 calls for clarity
                    caller_city = sim.user_locations[call['caller']]
                    callee_city = sim.user_locations[call['callee']]
                    
                    caller_coords = sim.city_coords[caller_city]
                    callee_coords = sim.city_coords[callee_city]
                    
                    call_data.append({
                        'caller': call['caller'],
                        'callee': call['callee'],
                        'lat1': caller_coords[0],
                        'lon1': caller_coords[1],
                        'lat2': callee_coords[0],
                        'lon2': callee_coords[1],
                        'latency': call['latency']
                    })
                
                call_df = pd.DataFrame(call_data)
                
                fig = go.Figure()
                
                # Add city markers
                for _, city in city_df.iterrows():
                    fig.add_trace(go.Scattermapbox(
                        lon=[city['lon']],
                        lat=[city['lat']],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        name=city['city'],
                        showlegend=False
                    ))
                
                # Add call lines
                for _, call in call_df.iterrows():
                    color = 'green' if call['latency'] < 3 else 'orange' if call['latency'] < 5 else 'red'
                    fig.add_trace(go.Scattermapbox(
                        mode='lines',
                        lon=[call['lon1'], call['lon2']],
                        lat=[call['lat1'], call['lat2']],
                        line=dict(width=1, color=color),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    mapbox_style="carto-positron",
                    mapbox_zoom=3,
                    mapbox_center_lat=37,
                    mapbox_center_lon=-95,
                    title="Call Patterns (Green=Low, Orange=Med, Red=High Latency)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No calls generated in this simulation run.")
    
    with tab2:
        st.header("Performance Metrics")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", sim.metrics['queries'])
        with col2:
            st.metric("Total Updates", sim.metrics['updates'])
        with col3:
            if sim.metrics['updates'] > 0:
                cmr = sim.metrics['queries'] / sim.metrics['updates']
                st.metric("Call-to-Mobility Ratio", f"{cmr:.2f}")
        with col4:
            total_calls = len(results['calls'])
            st.metric("Total Calls", total_calls)
        
        # Latency over time
        if sim.metrics['latency_history']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sim.metrics['latency_history'],
                mode='lines+markers',
                name='Average Latency',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Average Call Latency Over Time",
                xaxis_title="Time Step",
                yaxis_title="Average Latency (hops)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # CMR History
        if sim.metrics['cmr_history']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=sim.metrics['cmr_history'],
                mode='lines+markers',
                name='CMR',
                line=dict(color='orange', width=2)
            ))
            fig.update_layout(
                title="Call-to-Mobility Ratio Evolution",
                xaxis_title="Time Step",
                yaxis_title="CMR",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Network Hierarchy Visualization")
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌍 Regions: {num_regions}")
        with col2:
            st.info(f"🏙️ Total Cities: {num_regions * cities_per_region}")
        with col3:
            st.info(f"👥 Total Users: {num_regions * cities_per_region * users_per_city}")
        
        # Tree structure info
        st.subheader("Hierarchical Tree Structure")
        
        # Create tree visualization data
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
        
        # Movement patterns
        st.subheader("User Movement Patterns")
        movements_per_step = results['movements']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=movements_per_step,
            name='Users Moved',
            marker_color='purple'
        ))
        fig.update_layout(
            title="User Movements per Simulation Step",
            xaxis_title="Step",
            yaxis_title="Number of Users Moved"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Forwarding Pointer Analysis")
        
        # Forwarding hits by level
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forwarding Pointer Usage by Level")
            
            if sim.metrics['forwarding_hits']:
                fp_data = pd.DataFrame({
                    'Level': list(sim.metrics['forwarding_hits'].keys()),
                    'Hits': list(sim.metrics['forwarding_hits'].values())
                })
                
                fig = px.bar(
                    fp_data,
                    x='Level',
                    y='Hits',
                    title='Forwarding Pointer Hits',
                    color='Level',
                    color_discrete_map={'city': 'blue', 'region': 'green', 'root': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No forwarding pointer hits recorded.")
        
        with col2:
            st.subheader("Forwarding Effectiveness")
            
            # Calculate effectiveness metrics
            if results['calls']:
                levels_used = []
                for call in results['calls']:
                    if call['forwarding_levels']:
                        levels_used.extend(call['forwarding_levels'])
                
                if levels_used:
                    level_counts = pd.Series(levels_used).value_counts()
                    
                    fig = px.pie(
                        values=level_counts.values,
                        names=level_counts.index,
                        title='Distribution of Forwarding Level Usage'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No forwarding pointers used in calls.")
        
        # Latency distribution
        st.subheader("Latency Distribution Analysis")
        
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
                yaxis_title="Frequency",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Detailed Statistics")
        
        # Call statistics
        if results['calls']:
            st.subheader("Call Statistics")
            
            call_stats = pd.DataFrame(results['calls'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_latency = call_stats['latency'].mean()
                st.metric("Average Latency", f"{avg_latency:.2f} hops")
            
            with col2:
                max_latency = call_stats['latency'].max()
                st.metric("Maximum Latency", f"{max_latency} hops")
            
            with col3:
                min_latency = call_stats['latency'].min()
                st.metric("Minimum Latency", f"{min_latency} hops")
            
            # Show sample calls
            st.subheader("Sample Call Records")
            sample_calls = call_stats[['caller', 'callee', 'latency', 'optimal_level']].head(20)
            st.dataframe(sample_calls, use_container_width=True)
            
            # User activity analysis
            st.subheader("Most Active Users")
            
            call_frequency = {}
            for call in results['calls']:
                call_frequency[call['caller']] = call_frequency.get(call['caller'], 0) + 1
                call_frequency[call['callee']] = call_frequency.get(call['callee'], 0) + 1
            
            if call_frequency:
                top_users = pd.DataFrame({
                    'User': list(call_frequency.keys()),
                    'Call Count': list(call_frequency.values())
                }).sort_values('Call Count', ascending=False).head(10)
                
                fig = px.bar(
                    top_users,
                    x='User',
                    y='Call Count',
                    title='Top 10 Most Active Users'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Metrics to CSV"):
                metrics_df = pd.DataFrame({
                    'Metric': ['Total Queries', 'Total Updates', 'CMR'],
                    'Value': [
                        sim.metrics['queries'],
                        sim.metrics['updates'],
                        sim.metrics['queries'] / max(sim.metrics['updates'], 1)
                    ]
                })
                csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics CSV",
                    data=csv,
                    file_name='hierarchical_metrics.csv',
                    mime='text/csv'
                )
        
        with col2:
            if results['calls'] and st.button("📞 Export Calls to CSV"):
                calls_df = pd.DataFrame(results['calls'])
                csv = calls_df.to_csv(index=False)
                st.download_button(
                    label="Download Calls CSV",
                    data=csv,
                    file_name='hierarchical_calls.csv',
                    mime='text/csv'
                )

else:
    # Welcome screen
    st.info("👈 Configure simulation parameters in the sidebar and click 'Run Simulation' to start!")
    
    # Display information about the system
    st.markdown("""
    ### 📖 About This System
    
    This enhanced hierarchical location management system demonstrates:
    
    - **Multi-level Forwarding Pointers**: Implements forwarding at city, region, and root levels
    - **Realistic Mobility**: Distance-based movement probabilities
    - **Dynamic Optimization**: CMR-based level adjustment
    - **Performance Analysis**: Comprehensive metrics and visualizations
    
    ### 🎯 Key Features
    
    1. **Hierarchical Structure**: Root → Regions → Cities → Users
    2. **Smart Forwarding**: Reduces location query latency
    3. **Adaptive Strategy**: Adjusts to user behavior patterns
    4. **Visual Analytics**: Interactive maps and performance charts
    
    ### 🚀 Getting Started
    
    1. Adjust the network topology settings
    2. Configure user behavior parameters
    3. Set forwarding chain length
    4. Click 'Run Simulation' to see results!
    """)