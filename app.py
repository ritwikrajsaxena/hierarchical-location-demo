import streamlit as st
import plotly.express as px
from simulation import HierarchicalSimulator

st.set_page_config(layout="wide", page_title="Hierarchical Location Simulator")
st.title("Hierarchical Location Scheme Simulator (USA Map)")

# --- Sidebar Sliders ---
num_regions = st.sidebar.slider("Number of Regions", 2, 10, 4)
cities_per_region = st.sidebar.slider("Cities per Region", 2, 10, 5)
users_per_city = st.sidebar.slider("Users per City", 1, 20, 10)
mobility_prob = st.sidebar.slider("Mobility Probability", 0.0, 1.0, 0.1, 0.05)
call_prob = st.sidebar.slider("Call Probability", 0.0, 1.0, 0.3, 0.05)
forwarding_level = st.sidebar.slider("Forwarding Pointer Level", 0, 3, 1)
simulation_steps = st.sidebar.slider("Simulation Steps", 1, 10, 3)

# --- Run Simulation ---
sim = HierarchicalSimulator(num_regions=num_regions,
                             cities_per_region=cities_per_region,
                             users_per_city=users_per_city,
                             mobility_prob=mobility_prob,
                             call_prob=call_prob,
                             forwarding_level=forwarding_level)

call_df = sim.run_simulation(steps=simulation_steps)

st.write(f"Total Calls Simulated: {len(call_df)}")

# --- Map Plot ---
if not call_df.empty:
    fig = px.scatter_mapbox(
        call_df,
        lat="lat1",
        lon="lon1",
        hover_name="user1",
        hover_data=["city1", "latency"],
        color="latency",
        size_max=15,
        zoom=3,
        mapbox_style="carto-positron"
    )

    # Add lines for calls
    for _, row in call_df.iterrows():
        fig.add_trace(px.line_mapbox(
            lat=[row["lat1"], row["lat2"]],
            lon=[row["lon1"], row["lon2"]]
        ).data[0])

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No calls generated. Adjust probabilities or users to see calls.")

# --- Metrics ---
if not call_df.empty:
    st.write("### Latency Statistics")
    st.write(f"Average Latency: {call_df['latency'].mean():.2f}")
    st.write(f"Maximum Latency: {call_df['latency'].max()}")
    st.write(f"Minimum Latency: {call_df['latency'].min()}")