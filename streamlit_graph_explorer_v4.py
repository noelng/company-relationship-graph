"""
Streamlit Interactive Relationship Graph Visualization
Click nodes directly in the graph to expand their relationships

Install: pip install streamlit pyvis networkx pandas streamlit-javascript
Run: streamlit run streamlit_app.py

Note: For click-to-expand to work, this uses a polling mechanism.
After clicking a node in the graph, wait 1-2 seconds for it to expand.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import json
import time
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
if 'last_click_check' not in st.session_state:
    st.session_state.last_click_check = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

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
    if 'entity_type_from' in df.columns and 'entity_type_to' in df.columns:
        for _, row in df.iterrows():
            entity_types[row['entity_from']] = row.get('entity_type_from', 'company')
            entity_types[row['entity_to']] = row.get('entity_type_to', 'company')
    else:
        # Auto-detect based on common patterns
        for _, row in df.iterrows():
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
    company_keywords = ['inc', 'corp', 'ltd', 'llc', 'co.', 'company', 'group', 'holdings', 'pte']
    if any(keyword in name_lower for keyword in company_keywords):
        return 'company'
    
    # Default to company if uncertain
    return 'company'

# ============= Graph Visualization Functions =============

def get_relationship_color(relationship_sub_type):
    """Generate consistent color for each relationship sub-type"""
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
        'shareholder': '#f87171',
        'auditor': '#22d3ee',
        'multiple': '#000000',
        'owner': '#dc2626',
    }
    
    sub_type_lower = str(relationship_sub_type).lower().replace(' ', '_')
    if sub_type_lower in color_map:
        return color_map[sub_type_lower]
    
    hash_val = hash(relationship_sub_type) % 360
    return f'hsl({hash_val}, 70%, 50%)'

def get_node_icon(entity_type):
    """Get icon/shape based on entity type"""
    if entity_type == 'person':
        return {
            'shape': 'icon',
            'icon': {
                'face': 'FontAwesome',
                'code': '\uf007',
                'size': 50,
                'color': '#3b82f6'
            }
        }
    else:
        return {
            'shape': 'icon',
            'icon': {
                'face': 'FontAwesome',
                'code': '\uf1ad',
                'size': 50,
                'color': '#10b981'
            }
        }

def create_interactive_graph(G, entity_types, expanded_nodes, root_entity):
    """Generate interactive graph with click-to-expand functionality"""
    if not root_entity or root_entity not in G:
        return None, []
    
    nodes_to_show = set()
    edges_to_show = []
    available_to_expand = set()
    
    for node in expanded_nodes:
        if node in G:
            nodes_to_show.add(node)
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            nodes_to_show.update(neighbors)
            available_to_expand.update(neighbors - expanded_nodes)
            
            for neighbor in G.successors(node):
                edges_to_show.append((node, neighbor))
            for neighbor in G.predecessors(node):
                edges_to_show.append((neighbor, node))
    
    subgraph = nx.DiGraph()
    for edge in edges_to_show:
        if edge[0] in nodes_to_show and edge[1] in nodes_to_show:
            edge_data = G.get_edge_data(edge[0], edge[1])
            if edge_data:
                subgraph.add_edge(edge[0], edge[1], **edge_data)
    
    if root_entity not in subgraph:
        subgraph.add_node(root_entity)
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        notebook=False
    )
    
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        is_expanded = node in expanded_nodes
        is_root = node == root_entity
        can_expand = node in available_to_expand
        
        size = 35 if is_root else (28 if is_expanded else 25)
        
        if is_expanded:
            title_text = f"{node} ({entity_type})\n✓ Expanded"
        elif can_expand:
            title_text = f"{node} ({entity_type})\n👆 Click to expand"
        else:
            title_text = f"{node} ({entity_type})"
        
        net.add_node(
            node,
            label=node,
            title=title_text,
            **icon_config,
            size=size,
            borderWidth=3 if is_expanded else 1,
            borderWidthSelected=5
        )
    
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_type = edge_data.get('relationship_type', '')
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            title=f"{rel_type}: {rel_sub_type}",
            color=get_relationship_color(rel_sub_type),
            arrows='to',
            width=2,
            font={'size': 12, 'align': 'middle', 'strokeWidth': 0}
        )
    
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "stabilization": {
          "iterations": 300,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true,
        "tooltipDelay": 200
      },
      "edges": {
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.5
        }
      }
    }
    """)
    
    return net, sorted(list(available_to_expand))

def create_radial_map(G, entity_types, entity, depth=2):
    """Generate radial map with icons and directed edges"""
    if entity not in G:
        return None
    
    nodes_set = {entity}
    for d in range(depth):
        current_layer = list(nodes_set)
        for node in current_layer:
            nodes_set.update(G.successors(node))
            nodes_set.update(G.predecessors(node))
    
    subgraph = G.subgraph(nodes_set)
    
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        is_center = node == entity
        net.add_node(
            node,
            label=node,
            title=f"{node} ({entity_type})",
            **icon_config,
            size=35 if is_center else 25,
            borderWidth=3 if is_center else 1
        )
    
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_type = edge_data.get('relationship_type', '')
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            title=f"{rel_type}: {rel_sub_type}",
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
          "gravitationalConstant": -18000,
          "centralGravity": 0.3,
          "springLength": 180,
          "damping": 0.5
        },
        "stabilization": {
          "iterations": 250
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
    
    for node in subgraph.nodes():
        entity_type = entity_types.get(node, 'company')
        icon_config = get_node_icon(entity_type)
        
        is_selected = node in entities
        net.add_node(
            node,
            label=node,
            title=f"{node} ({entity_type})",
            **icon_config,
            size=30 if is_selected else 20,
            borderWidth=3 if is_selected else 1
        )
    
    for edge in subgraph.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        rel_sub_type = edge_data.get('relationship_sub_type', '')
        rel_type = edge_data.get('relationship_type', '')
        
        net.add_edge(
            edge[0],
            edge[1],
            label=rel_sub_type,
            title=f"{rel_type}: {rel_sub_type}",
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
          "gravitationalConstant": -15000,
          "centralGravity": 0.3,
          "springLength": 200
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
st.markdown("**Click directly on nodes in the graph to expand!** Use the buttons below as backup.")

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
            ["Interactive (Click Nodes)", "Radial Map", "Interconnection"]
        )
        
        if map_type == "Radial Map":
            depth = st.slider("Depth", 1, 5, 2)
        
        st.markdown("---")
        st.markdown("### 🎨 Legend")
        st.markdown("🏢 **Building** = Company")
        st.markdown("👤 **Person** = Individual")
        st.markdown("📍 **Thick border** = Expanded")
        st.markdown("*Edge colors = relationship sub-types*")

# Main content
if st.session_state.graph is None:
    st.info("👈 Upload a CSV file to begin")
    st.markdown("""
    ### CSV Format Required:
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
    
    if map_type == "Interconnection":
        st.subheader("Select Multiple Entities")
        selected_entities = st.multiselect(
            "Choose 2+ entities to find connections",
            entities,
            max_selections=5
        )
    else:
        st.subheader("🎯 Select Starting Entity")
        selected_entity = st.selectbox(
            "Choose an entity to start",
            [""] + entities,
            key="entity_selector"
        )
        
        if selected_entity and selected_entity != st.session_state.selected_entity:
            st.session_state.selected_entity = selected_entity
            st.session_state.expanded_nodes = {selected_entity}
        
        if map_type == "Interactive (Click Nodes)" and selected_entity:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.success("💡 **Click any node with '👆 Click to expand' in the graph to reveal its relationships!**")
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.expanded_nodes = {selected_entity}
                    st.rerun()
    
    # Generate visualization
    if map_type == "Interactive (Click Nodes)":
        if selected_entity:
            with st.spinner("Generating interactive graph..."):
                net, available_nodes = create_interactive_graph(G, entity_types, st.session_state.expanded_nodes, selected_entity)
                if net:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp:
                        net.save_graph(tmp.name)
                        tmp_path = tmp.name
                    
                    # Read and modify HTML
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Add click handling with cookie storage
                    expandable_json = json.dumps(available_nodes)
                    click_script = f"""
                    <script>
                    var expandableNodes = {expandable_json};
                    
                    network.on("click", function(params) {{
                        if (params.nodes.length > 0) {{
                            var nodeId = params.nodes[0];
                            if (expandableNodes.includes(nodeId)) {{
                                // Store in cookie
                                document.cookie = "expand_node=" + encodeURIComponent(nodeId) + "; path=/; max-age=5";
                                console.log("Node click stored:", nodeId);
                                
                                // Visual feedback
                                alert("Expanding: " + nodeId + "\\n\\nClick OK and wait 1-2 seconds for the graph to update.");
                            }}
                        }}
                    }});
                    
                    network.on("hoverNode", function(params) {{
                        if (expandableNodes.includes(params.node)) {{
                            document.body.style.cursor = 'pointer';
                        }}
                    }});
                    
                    network.on("blurNode", function() {{
                        document.body.style.cursor = 'default';
                    }});
                    </script>
                    """
                    html_content = html_content.replace('</body>', click_script + '</body>')
                    
                    # Display graph
                    st.components.v1.html(html_content, height=750, scrolling=False)
                    
                    # Check for clicks via cookie reader
                    cookie_reader = """
                    <script>
                    function getCookie(name) {
                        const value = `; ${document.cookie}`;
                        const parts = value.split(`; ${name}=`);
                        if (parts.length === 2) return parts.pop().split(';').shift();
                    }
                    
                    const expandNode = getCookie('expand_node');
                    if (expandNode) {
                        const decoded = decodeURIComponent(expandNode);
                        window.parent.postMessage({type: 'streamlit:setComponentValue', value: decoded}, '*');
                        document.cookie = "expand_node=; path=/; max-age=0";
                    }
                    </script>
                    """
                    
                    clicked_node = st.components.v1.html(cookie_reader, height=0)
                    
                    # If node was clicked, expand it
                    if clicked_node and clicked_node in available_nodes:
                        st.session_state.expanded_nodes.add(clicked_node)
                        st.rerun()
                    
                    # Stats
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Root Entity", selected_entity)
                    with col2:
                        st.metric("Incoming", G.in_degree(selected_entity))
                    with col3:
                        st.metric("Outgoing", G.out_degree(selected_entity))
                    with col4:
                        st.metric("Expanded", len(st.session_state.expanded_nodes))
                    
                    # Backup buttons
                    if available_nodes:
                        with st.expander("🔍 Manual Expand (Backup Method)", expanded=False):
                            st.info("If clicking nodes in the graph doesn't work, use these buttons:")
                            num_cols = 4
                            cols = st.columns(num_cols)
                            
                            for idx, node in enumerate(available_nodes):
                                with cols[idx % num_cols]:
                                    entity_type = entity_types.get(node, 'company')
                                    icon = "👤" if entity_type == "person" else "🏢"
                                    if st.button(f"{icon} {node}", key=f"expand_{node}", use_container_width=True):
                                        st.session_state.expanded_nodes.add(node)
                                        st.rerun()
                    
                    # Show expanded list
                    if len(st.session_state.expanded_nodes) > 1:
                        with st.expander("✅ Currently Expanded Entities", expanded=False):
                            for node in sorted(st.session_state.expanded_nodes):
                                entity_type = entity_types.get(node, 'company')
                                icon = "👤" if entity_type == "person" else "🏢"
                                st.write(f"{icon} **{node}**")
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
