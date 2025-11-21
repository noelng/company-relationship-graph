"""
Streamlit Quick Prototype for Relationship Graph Visualization
Single-file implementation for rapid testing

Install: pip install streamlit pyvis networkx pandas
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
from pathlib import Path

st.set_page_config(page_title="Relationship Graph Explorer", layout="wide")

# ============= Initialize Session State =============
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'entities' not in st.session_state:
    st.session_state.entities = []

# ============= Data Loading =============
@st.cache_data
def load_graph_from_csv(csv_file):
    """Load CSV and create NetworkX graph"""
    df = pd.read_csv(csv_file)
    
    # Validate columns
    required_cols = ['entity_from', 'relationship_type', 'relationship_sub_type', 'entity_to']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain: {required_cols}")
        return None, []
    
    # Clean data
    df = df.dropna(subset=['entity_from', 'entity_to'])
    df['entity_from'] = df['entity_from'].str.strip()
    df['entity_to'] = df['entity_to'].str.strip()
    
    # Build graph
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(
            row['entity_from'],
            row['entity_to'],
            relationship_type=row['relationship_type'],
            relationship_sub_type=row.get('relationship_sub_type', '')
        )
    
    entities = sorted(list(G.nodes()))
    
    return G, entities

# ============= Graph Visualization Functions =============

def create_radial_map(G, entity, depth=2):
    """Generate radial map using PyVis"""
    if entity not in G:
        return None
    
    # Get subgraph
    nodes_set = {entity}
    for d in range(depth):
        current_layer = list(nodes_set)
        for node in current_layer:
            nodes_set.update(G.successors(node))
            nodes_set.update(G.predecessors(node))
    
    subgraph = G.subgraph(nodes_set)
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(subgraph)
    
    # Customize appearance
    for node in net.nodes:
        node['color'] = '#3b82f6' if node['id'] != entity else '#ef4444'
        node['size'] = 30 if node['id'] == entity else 20
    
    # Color edges by type
    color_map = {
        'ownership': '#ef4444',
        'investment': '#3b82f6',
        'control': '#8b5cf6',
        'employment': '#10b981'
    }
    
    for edge in net.edges:
        edge_data = G.get_edge_data(edge['from'], edge['to'])
        rel_type = edge_data.get('relationship_type', '').lower()
        edge['color'] = color_map.get(rel_type, '#999999')
        edge['title'] = f"{rel_type}: {edge_data.get('relationship_sub_type', '')}"
    
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 200
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    return net

def create_lbo_map(G, entity, max_depth=10):
    """Beneficial ownership map - trace owners recursively"""
    if entity not in G:
        return None
    
    ownership_types = {'ownership', 'control', 'owns', 'controls'}
    
    def trace_owners(node, visited, depth):
        if depth > max_depth or node in visited:
            return visited
        visited.add(node)
        
        for predecessor in G.predecessors(node):
            edge_data = G[predecessor][node]
            rel_type = edge_data.get('relationship_type', '').lower()
            if any(own in rel_type for own in ownership_types):
                trace_owners(predecessor, visited, depth + 1)
        
        return visited
    
    ownership_chain = trace_owners(entity, set(), 0)
    subgraph = G.subgraph(ownership_chain)
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    net.from_nx(subgraph)
    
    for node in net.nodes:
        node['color'] = '#ef4444' if node['id'] == entity else '#fbbf24'
        node['size'] = 30 if node['id'] == entity else 20
    
    return net

def create_investee_map(G, entity, depth=2):
    """Show entities the subject invests in"""
    if entity not in G:
        return None
    
    investment_types = {'investment', 'ownership', 'owns', 'invests'}
    
    nodes_set = {entity}
    current_layer = {entity}
    
    for d in range(depth):
        next_layer = set()
        for node in current_layer:
            for successor in G.successors(node):
                edge_data = G[node][successor]
                rel_type = edge_data.get('relationship_type', '').lower()
                if any(inv in rel_type for inv in investment_types):
                    next_layer.add(successor)
                    nodes_set.add(successor)
        current_layer = next_layer
    
    subgraph = G.subgraph(nodes_set)
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    net.from_nx(subgraph)
    
    for node in net.nodes:
        node['color'] = '#ef4444' if node['id'] == entity else '#10b981'
        node['size'] = 30 if node['id'] == entity else 20
    
    return net

def create_interconnection_map(G, entities):
    """Find paths between multiple entities"""
    all_nodes = set()
    
    for i, source in enumerate(entities):
        for target in entities[i+1:]:
            if source in G and target in G:
                try:
                    paths = list(nx.all_simple_paths(G.to_undirected(), source, target, cutoff=5))
                    for path in paths[:3]:  # Limit to 3 paths per pair
                        all_nodes.update(path)
                except nx.NetworkXNoPath:
                    continue
    
    if not all_nodes:
        return None
    
    subgraph = G.subgraph(all_nodes)
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    net.from_nx(subgraph)
    
    for node in net.nodes:
        node['color'] = '#ef4444' if node['id'] in entities else '#3b82f6'
        node['size'] = 30 if node['id'] in entities else 20
    
    return net

# ============= Streamlit UI =============

st.title("🔗 Relationship Graph Explorer")
st.markdown("Interactive network visualization tool for exploring entity relationships")

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        G, entities = load_graph_from_csv(uploaded_file)
        if G:
            st.session_state.graph = G
            st.session_state.entities = entities
            st.success(f"✓ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    st.markdown("---")
    
    if st.session_state.graph:
        st.header("📊 Graph Stats")
        G = st.session_state.graph
        st.metric("Total Nodes", G.number_of_nodes())
        st.metric("Total Edges", G.number_of_edges())
        st.metric("Density", f"{nx.density(G):.4f}")
        
        st.markdown("---")
        st.header("🎨 Visualization Options")
        
        map_type = st.radio(
            "Map Type",
            ["Radial Map", "LBO (Beneficial Ownership)", "Investee Map", "Interconnection"]
        )
        
        if map_type in ["Radial Map", "Investee Map"]:
            depth = st.slider("Depth", 1, 5, 2)
        
        st.markdown("---")
        
        # Legend
        st.markdown("### 🎨 Edge Colors")
        st.markdown("🔴 Ownership")
        st.markdown("🔵 Investment")
        st.markdown("🟣 Control")
        st.markdown("🟢 Employment")

# Main content
if st.session_state.graph is None:
    st.info("👈 Upload a CSV file to begin")
    st.markdown("""
    ### CSV Format Required:
    - `entity_from`: Source entity name
    - `relationship_type`: Type (ownership, investment, etc.)
    - `relationship_sub_type`: Subtype details
    - `entity_to`: Target entity name
    
    ### Example:
    ```csv
    entity_from,relationship_type,relationship_sub_type,entity_to
    Acme Corp,ownership,direct_ownership,Subsidiary Inc
    John Doe,employment,ceo,Acme Corp
    ```
    """)
else:
    G = st.session_state.graph
    entities = st.session_state.entities
    
    # Entity selection
    if map_type == "Interconnection":
        st.subheader("Select Multiple Entities")
        selected_entities = st.multiselect(
            "Choose 2+ entities to find connections",
            entities,
            max_selections=5
        )
    else:
        st.subheader("Select Entity")
        selected_entity = st.selectbox("Choose an entity", [""] + entities)
    
    # Generate visualization
    if map_type == "Interconnection":
        if len(selected_entities) >= 2:
            with st.spinner("Generating interconnection map..."):
                net = create_interconnection_map(G, selected_entities)
                if net:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                        net.save_graph(tmp.name)
                        st.components.v1.html(open(tmp.name, 'r').read(), height=650)
                else:
                    st.warning("No paths found between selected entities")
        else:
            st.info("Select at least 2 entities")
    
    elif selected_entity:
        with st.spinner(f"Generating {map_type.lower()}..."):
            if map_type == "Radial Map":
                net = create_radial_map(G, selected_entity, depth)
            elif map_type == "LBO (Beneficial Ownership)":
                net = create_lbo_map(G, selected_entity)
            elif map_type == "Investee Map":
                net = create_investee_map(G, selected_entity, depth)
            
            if net:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                    net.save_graph(tmp.name)
                    st.components.v1.html(open(tmp.name, 'r').read(), height=650)
                
                # Entity details
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Incoming Connections", G.in_degree(selected_entity))
                with col2:
                    st.metric("Outgoing Connections", G.out_degree(selected_entity))
                with col3:
                    st.metric("Total Degree", G.degree(selected_entity))
            else:
                st.error(f"Entity '{selected_entity}' not found in graph")
