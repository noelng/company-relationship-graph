"""
Streamlit Interactive Relationship Graph Visualization
Click to expand node relationships dynamically

Install: pip install streamlit pyvis networkx pandas
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import json
from pathlib import Path

st.set_page_config(page_title="Relationship Graph Explorer", layout="wide")

# ============= Initialize Session State =============
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'entities' not in st.session_state:
    st.session_state.entities = []
if 'entity_types' not in st.session_state:
    st.session_state.entity_types = {}
if 'expanded_nodes' not in st.session_state:
    st.session_state.expanded_nodes = set()
if 'selected_entity' not in st.session_state:
    st.session_state.selected_entity = None

# ============= Data Loading =============
@st.cache_data
def load_graph_from_csv(csv_file):
    """Load CSV and create NetworkX graph"""
    df = pd.read_csv(csv_file)
    
    # Validate columns
    required_cols = ['entity_from', 'relationship_type', 'relationship_sub_type', 'entity_to']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain: {required_cols}")
        return None, [], {}
    
    # Clean data
    df = df.dropna(subset=['entity_from', 'entity_to'])
    df['entity_from'] = df['entity_from'].str.strip()
    df['entity_to'] = df['entity_to'].str.strip()
    
    # Build graph
    G = nx.DiGraph()
    entity_types = {}
    
    # Detect entity types (check if 'entity_type' column exists)
    if 'entity_type' in df.columns:
        for _, row in df.iterrows():
            entity_types[row['entity_from']] = row.get('entity_type_from', 'company')
            entity_types[row['entity_to']] = row.get('entity_type_to', 'company')
    else:
        # Auto-detect based on common patterns
        for _, row in df.iterrows():
            # Simple heuristic: if name contains common person indicators
            entity_types[row['entity_from']] = detect_entity_type(row['entity_from'])
            entity_types[row['entity_to']] = detect_entity_type(row['entity_to'])
    
    for _, row in df.iterrows():
        G.add_edge(
            row['entity_from'],
            row['entity_to'],
            relationship_type=row['relationship_type'],
            relationship_sub_type=row.get('relationship_sub_type', '')
        )
    
    entities = sorted(list(G.nodes()))
    
    return G, entities, entity_types

def detect_entity_type(name):
    """Simple heuristic to detect if entity is person or company"""
    person_keywords = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
    name_lower = name.lower()
    
    # Check for person indicators
    if any(keyword in name_lower for keyword in person_keywords):
        return 'person'
    
    # Check for company indicators
    company_keywords = ['inc', 'corp', 'ltd', 'llc', 'co.', 'company', 'group', 'holdings']
    if any(keyword in name_lower for keyword in company_keywords):
        return 'company'
    
    # Default to company if uncertain
    return 'company'

# ============= Graph Visualization Functions =============

def get_relationship_color(relationship_sub_type):
    """Generate consistent color for each relationship sub-type"""
    # Predefined colors for common relationship types
    color_map = {
        'direct_ownership': '#ef4444',
        'indirect_ownership': '#f97316',
        'majority_shareholder': '#dc2626',
        'minority_shareholder': '#fb923c',
        'ceo': '#8b5cf6',
        'cfo': '#a78bfa',
        'director': '#c084fc',
        'board_member': '#d8b4fe',
        'employee': '#10b981',
        'consultant': '#34d399',
        'investment': '#3b82f6',
        'venture_capital': '#60a5fa',
        'private_equity': '#2563eb',
        'control': '#ec4899',
        'subsidiary': '#f43f5e',
    }
    
    # Return predefined color or generate hash-based color
    sub_type_lower = str(relationship_sub_type).lower().replace(' ', '_')
    if sub_type_lower in color_map:
        return color_map[sub_type_lower]
    
    # Generate color from hash
    hash_val = hash(relationship_sub_type) % 360
    return f'hsl({hash_val}, 70%, 50%)'

def get_node_icon(entity_type):
    """Get icon/shape based on entity type"""
    if entity_type == 'person':
        return {
            'shape': 'icon',
            'icon': {
                'face': 'FontAwesome',
                'code': '\uf007',  # fa-user
                'size': 50,
                'color': '#3b82f6'
            }
        }
    else:  # company
        return {
            'shape': 'icon',
            'icon': {
                'face': 'FontAwesome',
                'code': '\uf1ad',  # fa-building
                'size': 50,
                'color': '#10b981'
            }
        }

def create_expandable_graph(G, entity_types, expanded_nodes, root_entity=None):
    """Generate interactive expandable graph"""
    if not root_entity:
        return None
    
    # Start with root entity and expanded nodes
    nodes_to_show = expanded_nodes.copy()
    nodes_to_show.add(root_entity)
    
    # Get all edges between visible nodes
    edges_to_show = []
    for node in nodes_to_show:
        if node in G:
            # Outgoing edges
            for successor in G.successors(node):
                if successor in nodes_to_show:
                    edges_to_show.append((node, successor))
    
    # Create subgraph
    subgraph = nx.DiGraph()
    for edge in edges_to_show:
        edge_data = G.get_edge_data(edge[0], edge[1])
        subgraph.add_edge(edge[0], edge[1], **edge_data)
    
    # Add isolated root if needed
    if root_entity not in subgraph:
        subgraph.add_node(root_entity)
    
    # Create PyVis network
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Add nodes with icons
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        net.add_node(
            node,
            label=node,
            title=f"{node} ({entity_type})\nClick to expand",
            **icon_config,
            size=30 if node == root_entity else 25
        )
    
    # Add edges with labels and colors
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_type = edge_data.get('relationship_type', '')
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        
        color = get_relationship_color(rel_sub_type)
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            title=f"{rel_type}: {rel_sub_type}",
            color=color,
            arrows='to',
            width=2,
            font={'size': 12, 'align': 'middle'}
        )
    
    # Set physics options
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -15000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        },
        "stabilization": {
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal"
        }
      }
    }
    """)
    
    return net

def create_radial_map(G, entity_types, entity, depth=2):
    """Generate radial map with icons and directed edges"""
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
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Add nodes with icons
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        is_center = node == entity
        net.add_node(
            node,
            label=node,
            title=f"{node} ({entity_type})",
            **icon_config,
            size=35 if is_center else 25
        )
    
    # Add edges with labels and colors
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_type = edge_data.get('relationship_type', '')
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        
        color = get_relationship_color(rel_sub_type)
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            title=f"{rel_type}: {rel_sub_type}",
            color=color,
            arrows='to',
            width=2,
            font={'size': 12, 'align': 'middle'}
        )
    
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -15000,
          "centralGravity": 0.3,
          "springLength": 150
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      }
    }
    """)
    
    return net

def create_interconnection_map(G, entity_types, entities):
    """Find paths between multiple entities with icons"""
    all_nodes = set()
    
    for i, source in enumerate(entities):
        for target in entities[i+1:]:
            if source in G and target in G:
                try:
                    paths = list(nx.all_simple_paths(G.to_undirected(), source, target, cutoff=5))
                    for path in paths[:3]:
                        all_nodes.update(path)
                except nx.NetworkXNoPath:
                    continue
    
    if not all_nodes:
        return None
    
    subgraph = G.subgraph(all_nodes)
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        directed=True
    )
    
    # Add nodes with icons
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        is_selected = node in entities
        net.add_node(
            node,
            label=node,
            title=f"{node} ({entity_type})",
            **icon_config,
            size=30 if is_selected else 20
        )
    
    # Add edges
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            color=get_relationship_color(rel_sub_type),
            arrows='to',
            width=2,
            font={'size': 12, 'align': 'middle'}
        )
    
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.3,
          "springLength": 180
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      }
    }
    """)
    
    return net

# ============= Streamlit UI =============

st.title("🔗 Relationship Graph Explorer")
st.markdown("Interactive network visualization - **Click nodes to expand relationships**")

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        G, entities, entity_types = load_graph_from_csv(uploaded_file)
        if G:
            st.session_state.graph = G
            st.session_state.entities = entities
            st.session_state.entity_types = entity_types
            st.success(f"✓ Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    st.markdown("---")
    
    if st.session_state.graph:
        st.header("📊 Graph Stats")
        G = st.session_state.graph
        st.metric("Total Nodes", G.number_of_nodes())
        st.metric("Total Edges", G.number_of_edges())
        st.metric("Density", f"{nx.density(G):.4f}")
        
        st.markdown("---")
        st.header("🎨 Visualization Mode")
        
        map_type = st.radio(
            "Select Mode",
            ["Interactive Expansion", "Radial Map", "Interconnection"]
        )
        
        if map_type == "Radial Map":
            depth = st.slider("Depth", 1, 5, 2)
        
        if map_type == "Interactive Expansion" and st.session_state.expanded_nodes:
            if st.button("🔄 Reset Expansion"):
                st.session_state.expanded_nodes = set()
                if st.session_state.selected_entity:
                    st.session_state.expanded_nodes.add(st.session_state.selected_entity)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎨 Legend")
        st.markdown("🏢 Building = Company")
        st.markdown("👤 Person = Individual")
        st.markdown("*Edge colors vary by relationship sub-type*")

# Main content
if st.session_state.graph is None:
    st.info("👈 Upload a CSV file to begin")
    st.markdown("""
    ### CSV Format Required:
    - `entity_from`: Source entity name
    - `relationship_type`: Type (ownership, investment, etc.)
    - `relationship_sub_type`: Subtype details
    - `entity_to`: Target entity name
    - `entity_type_from` (optional): 'person' or 'company'
    - `entity_type_to` (optional): 'person' or 'company'
    
    ### Example:
    ```csv
    entity_from,relationship_type,relationship_sub_type,entity_to,entity_type_from,entity_type_to
    Acme Corp,ownership,direct_ownership,Subsidiary Inc,company,company
    John Doe,employment,ceo,Acme Corp,person,company
    ```
    """)
else:
    G = st.session_state.graph
    entities = st.session_state.entities
    entity_types = st.session_state.entity_types
    
    # Entity selection
    if map_type == "Interconnection":
        st.subheader("Select Multiple Entities")
        selected_entities = st.multiselect(
            "Choose 2+ entities to find connections",
            entities,
            max_selections=5
        )
    elif map_type == "Interactive Expansion":
        st.subheader("🎯 Select Starting Entity")
        selected_entity = st.selectbox(
            "Choose an entity to start exploring",
            [""] + entities,
            key="entity_selector"
        )
        
        if selected_entity and selected_entity != st.session_state.selected_entity:
            st.session_state.selected_entity = selected_entity
            st.session_state.expanded_nodes = {selected_entity}
            
        if selected_entity:
            st.info("💡 **How to use:** The graph shows your selected entity and its direct relationships. Click on any node in the visualization below to expand and reveal its connections!")
            
            # Show expand buttons for connected entities
            if selected_entity in G:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Connected entities (click to expand):**")
                    connected = set(G.successors(selected_entity)) | set(G.predecessors(selected_entity))
                    unexpanded = connected - st.session_state.expanded_nodes
                    
                    if unexpanded:
                        for entity in sorted(list(unexpanded))[:10]:
                            if st.button(f"➕ {entity}", key=f"expand_{entity}"):
                                st.session_state.expanded_nodes.add(entity)
                                st.rerun()
                    else:
                        st.write("All connected entities shown")
                
                with col2:
                    if len(st.session_state.expanded_nodes) > 1:
                        st.markdown("**Expanded entities:**")
                        for entity in sorted(list(st.session_state.expanded_nodes - {selected_entity})):
                            st.write(f"✓ {entity}")
    else:
        st.subheader("Select Entity")
        selected_entity = st.selectbox("Choose an entity", [""] + entities)
    
    # Generate visualization
    if map_type == "Interactive Expansion":
        if selected_entity:
            with st.spinner("Generating interactive graph..."):
                net = create_expandable_graph(G, entity_types, st.session_state.expanded_nodes, selected_entity)
                if net:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
                        net.save_graph(tmp.name)
                        with open(tmp.name, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=750)
                    
                    # Entity details
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Incoming Connections", G.in_degree(selected_entity))
                    with col2:
                        st.metric("Outgoing Connections", G.out_degree(selected_entity))
                    with col3:
                        st.metric("Expanded Nodes", len(st.session_state.expanded_nodes))
        else:
            st.info("Select an entity to begin exploring")
    
    elif map_type == "Interconnection":
        if len(selected_entities) >= 2:
            with st.spinner("Generating interconnection map..."):
                net = create_interconnection_map(G, entity_types, selected_entities)
                if net:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
                        net.save_graph(tmp.name)
                        with open(tmp.name, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=750)
                else:
                    st.warning("No paths found between selected entities")
        else:
            st.info("Select at least 2 entities")
    
    elif selected_entity:
        with st.spinner(f"Generating {map_type.lower()}..."):
            if map_type == "Radial Map":
                net = create_radial_map(G, entity_types, selected_entity, depth)
            
            if net:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as tmp:
                    net.save_graph(tmp.name)
                    with open(tmp.name, 'r') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=750)
                
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
