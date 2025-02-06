'''
This file contains the functions required for the initial evacuation algorithm 
as well as a single update the initial evauation plan. 
'''
#### Imports
import numpy as np
import networkx as nx
import math
import time
from scipy.interpolate import UnivariateSpline
import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import transform
from shapely.ops import unary_union
from networkx.algorithms.flow import shortest_augmenting_path
import geopandas as gpd
import contextily as cx

#### Functions for Buildling the Original Road Network Object
MOORE_AFTER_BREAK_SPLINE = UnivariateSpline(
    [20, 30, 40, 50, 60, 70, 80, 90, 100],
    [3.9, 6, 11, 18, 27, 39, 54, 58, 84],
)
MOORE_BEFORE_BREAK_SPLINE = UnivariateSpline(
    [20, 30, 40, 50, 60, 70, 80, 90, 100],
    [6, 8, 11, 14, 17, 19, 22, 25, 28],
)

MOORE_SAFE_BREAKING_DISTANCE = lambda x: MOORE_AFTER_BREAK_SPLINE(
    x
) + MOORE_BEFORE_BREAK_SPLINE(x)

def moore(lanes: float, max_speed: float):
    return 1000 * max_speed / MOORE_SAFE_BREAKING_DISTANCE(max_speed) * lanes

def add_capacities(G, method=moore):
    '''
    Inputs: 
    G: a Networkx MultiDigraph
    method: the method with which to calculate road capacity; the default method is the moore method

    Output:
    returns the MultiDigraph with the edge attribute "upper" for all edges
    '''

    G = G.copy()
    cap = []
    
    for u, v, i in G.edges:
        edge_data = G.get_edge_data(u, v, i)
        raw_lanes = edge_data.get("lanes")
        
        if raw_lanes is None:
            lanes = 1
            
        elif isinstance(raw_lanes, str):
            lanes = int(raw_lanes) / 2  
            
        elif isinstance(raw_lanes, list):
            lanes = sum(int(x) for x in raw_lanes) / 2
            
        G[u][v][i]["upper"] = int(method(lanes, edge_data["speed_kph"]))
    return G

def add_orig_dest(G, sup_dem:[int, np.ndarray]):
    '''
    Inputs:
    G: a NetworkX MultiDiGraph
    sup_dem: an numpy array of tuples of the form (i,b_i) where i is a node in G and b_i is the supply/demand value for node i

    Output:
    returns the MultiDiGraph G with the node attribute "sup_dem" for all nodes
    '''
    if type(sup_dem) == int:
        for i in G.nodes:
            G._node[i]['sup_dem'] = sup_dem
    else:
        for i in G.nodes:
            G._node[i]['sup_dem'] = sup_dem[i]
    return G

def construct_orig_graph(location, sup_dem_nodes, **kwargs):
    '''
    Inputs:
    location: string with name of city/county/state of place you would like the MultiDiGraph road network of or tuple of the form
              (lat_1, long_1, lat_2, long_2) of lat and long coordinates for a bounding box around location you would like like the 
              MultiDiGraph road network of or tuple of the form (lat_1, long_1) which corresponds to the center of a circle of which 
              you would like like the MultiDiGraph road network of
    sup_dem_nodes: a numpy array of tuples of the form (i, b_i) where i is a node in G and b_i is the supply/demand value for node i

    optional inputs:
    distance: integer, length in meters of the radius for the circle associated with locations of the form (lat_1, long_1); default size of 1,000
    tolerance: integer, distance in meters of nodes in MultiDiGraph which should be contracted; default value of 10
    verbose: boolean for including print statements or not; default value is False

    Output:
    returns a MultiDiGraph G which includes the added node attribute "sup_dem" and edge attributes "speed", "travel_time",and "upper"
    '''

    distance = kwargs.get('distance',1000)
    tolerance_val = kwargs.get('tolerance',10)
    output = kwargs.get('verbose', False)
    
    start = time.time()
    #import data from osmnx, can input any city, state, etc. 
    if type(location) == str:
        G = ox.project_graph(ox.graph_from_place(location, network_type='drive'))
    elif type(location) == tuple:
        G = ox.graph_from_bbox(location[0],location[1],location[2],location[3], network_type='drive')
        G = ox.project_graph(G)
    else:
        G = ox.graph_from_point((location['lat'][0], location['lon'][0]), network_type='drive', dist = distance)
        G = ox.project_graph(G)
    if output:
        print('Importing Network Complete')

    #get rid of intersections that are not actually intersections
    G = ox.simplification.consolidate_intersections(G, tolerance=tolerance_val, rebuild_graph=True, dead_ends=True, reconnect_edges = True)
    if output:
        print('Consolidation Complete')

    #add edge speeds
    G = ox.add_edge_speeds(G)
    if output:
        print('Speed Added')

    #add travel times
    G = ox.add_edge_travel_times(G)
    if output:
        print('Travel Time Added')

    #add capacities (computed using moore method)
    G = add_capacities(G)
    if output:
        print('Capacities Added')

    sup_dem = np.zeros(len(G.nodes))
    for i,j in sup_dem_nodes:
        sup_dem[i] = j
    G = add_orig_dest(G,sup_dem)
    if output:
        print('Supply and Demand Values Added')

    end = time.time()

    if output:
        print('Time to Build Original Network: ',end-start, ' seconds')
    
    return G

#### Functions for the Fire Polygon
def geodesic_point_buffer(G, lat, lon, radius):
    """
    Inputs:
    G: NetworkX MultiDiGraph object of road network 
    lat: latitude for center of circle
    lon: longitude for center of circle
    radius: radius of circle in meters

    Output:
    returns a circle from given latitude, longitude, and radius in the CRS of G
    """
    aeqd_proj = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0")
    tfmr = Transformer.from_proj(aeqd_proj, aeqd_proj.geodetic_crs)
    buf = Point(0, 0).buffer(radius * 1609.34)  # distance in miles (converts from meters to miles)
    circle = Polygon(transform(tfmr.transform, buf).exterior.coords[:])
    net_trans = Transformer.from_crs(aeqd_proj.geodetic_crs, G.graph['crs'])
    proj_circ = Polygon(transform(net_trans.transform,circle).exterior.coords[:])
    
    return proj_circ

def dist_to_fire(G, fire_poly):
    '''
    Inputs:
    G: NetworkX MultiDiGraph object of road network
    fire_poly: MultiPolygon or Polygon object which represents the fire

    Outputs:
    nodes_removed: list of nodes which intersect fire_poly
    edge_dist: dictionary of edges with their distance to fire poly

    '''
    ###get geometric objects from G
    gs_nodes = ox.utils_graph.graph_to_gdfs(G,nodes = True,edges = False, node_geometry = True)[['geometry']]
    gs_edges = ox.utils_graph.graph_to_gdfs(G,nodes = False, edges = True)[['geometry']]

    ###determine which nodes intersect fire_poly and distance of every edge to fire_poly
    nodes_removed = []
    edge_dist = {}

    for point in gs_nodes['geometry']:
        if point.intersects(fire_poly):
            node = ox.distance.nearest_nodes(G,point.x,point.y,return_dist=False)
            nodes_removed.append(node)
    for i in range(len(gs_edges)):
        entry =  gs_edges.iloc[i]
        dist = entry['geometry'].distance(fire_poly)
        entry = gs_edges.iloc[[i]]
        edge = entry['geometry'].index[0]
        if G.is_multigraph():
            edge_dist[(edge[0],edge[1],edge[2])] = dist
        elif G.is_directed():
            edge_dist[(edge[0],edge[1])] = dist
        else:
            raise Exception("Must Input a DiGraph or MultiDiGraph")

    return (nodes_removed, edge_dist)

def create_fire_mat(G, fire_orig_radii, T, removed_node_mat, edge_dist_mat, fire_poly_mat,**kwargs):
    '''
    Inputs:
    G: NetworkX MultiDiGraph object of road network
    fire_orig_radii: list of tuples of the form (lat, long, init_radius) for the circles of the fire polygon at a time t
    T: time horizon of a time-expanded network
    removed_node_mat: list of lists which contain the nodes which intersect the fire polygon at times t = 0,...,T
    edge_dist_mat: list of lists which contain the the distance of edges to the fire polygon at times t = 0,...,T
    fire_poly_mat: list of lists whcih contain MultiPolygon or Polygons representing a fire polygon at times t = 0,...,T

    optional inputs:
    start_time_int: time t for which you would like to begin creating fire polygons for fire_poly_mat; default is len(removed_node_mat)
    verbose: option to include print statements; defulat is False

    Outputs:
    Returns updated lists of lists for the following objects for all t = 0,...,T (note that T may have changed and therefore more 
    information needed to be added to the previously exists lists of lists)

    removed_node_mat_copy: new removed_node_mat with extra time intervals
    edge_dist_mat_copy: new removed_node_mat with extra time intervals
    fire_polygon_mat_copy: new removed_node_mat with extra time intervals
    '''
    removed_node_mat_copy = removed_node_mat.copy()
    edge_dist_mat_copy = edge_dist_mat.copy()
    fire_polygon_mat_copy = fire_poly_mat.copy()
    
    start = time.time()
    time_ints_done = kwargs.get('start_time_int',len(removed_node_mat))
    output = kwargs.get('verbose', False)
    

    for i in range(time_ints_done, T):
        fire_polys = []
        for entry in fire_orig_radii:
            fire_lat = entry[0]
            fire_long = entry[1]
            init_radius = entry[2]
        
            ###create fire_poly in coordinate system of network
            radius = init_radius+(0.005*i)
            fire_poly_piece = geodesic_point_buffer(G,fire_lat,fire_long,radius)
            fire_polys.append(fire_poly_piece)
        
        ###create MultiPolygon or Polygon obnect from union of circles
        fire_polygon = unary_union(fire_polys)
        ###3 deteremine nodes which have been overtaken by fire and distance from edges to fire
        nodes_removed, edge_distances = dist_to_fire(G, fire_polygon)

        ###update previous lists of lists
        fire_polygon_mat_copy.append(fire_polygon)
        removed_node_mat_copy.append(nodes_removed)
        edge_dist_mat_copy.append(edge_distances)
        
    end = time.time()
    if output:
        print(f"Time to do Fire Poly for Time Horizon {T-1}: {end - start} seconds")
    
    return (removed_node_mat_copy, edge_dist_mat_copy, fire_polygon_mat_copy)

#### Function for the Creation of the Time-expanded Network
def time_expand_with_removal_dyn(G, ten, time_int_size, prev_T, curr_T,**kwargs):
    '''
    Inputs:
    G: NetworkX DiGraph or MultiDiGraph object
    ten: Networkx DiGraph which represents a time-expanded netwrok
    time_int_size: integer, interval length for time instances (ex: 1 means each time step is a minute, 2 meanns each time step is 2 minutes)
    prev_T: integer, previous time horizon T
    curr_T: integer, current time horizon T 

    optional inputs:
    removed_nodes_mat: list of lists which represent the nodes removed at each time instance; default is empy lists of lists
    edge_distance_mat: list of lists which contains the distance from every edge to the fire for each time instance; default is empy lists of lists
    verbose: boolean for including print statements; default is False

    Outputs:
    Returns a time-expanded network (ten_copy) with time horizon curr_T which has incorporated fire data into its construction 
    '''
    start = time.time()
    orig_nodes = list(G.nodes)
    orig_edges = list(G.edges)
    ten_copy = ten.copy()
    percent_cap = 1
    removed_nodes_mat = kwargs.get('removed_nodes_mat',[ [] for _ in range(curr_T)])
    edge_dist_mat = kwargs.get('edge_distance_mat',[ {} for _ in range(curr_T)])
    output = kwargs.get('verbose', False)
    lb_percent = kwargs.get('lb_percent', 0.2)


    ###add in all nodes for time-expanded network
    if prev_T == 0:
        node_num = 1
    else:
        node_num = len(orig_nodes)*prev_T+1
            
    for j in range(prev_T,curr_T):
        for i in range(len(orig_nodes)):
            #check that i is not one of nodes that have been removed during that time interval
            if i not in removed_nodes_mat[j]:
                if j == 0:
                    node_name = f'{i+1}-{j}'
                    ten_copy.add_node(node_num, name = node_name, level = j, sup_dem = G._node[i]['sup_dem'])
                else:
                    node_name = f'{i+1}-{j}'
                    ten_copy.add_node(node_num, name = node_name, level = j, sup_dem = 0)
            node_num += 1
            
    ###add in hold over arcs for supply/demand nodes
    if prev_T == 0:
        start_time_int = prev_T
    else:
        start_time_int = prev_T-1
    for i in orig_nodes:
        if G._node[i]['sup_dem'] >  0:
            for j in range(start_time_int,curr_T-1):
                if i not in removed_nodes_mat[j+1]:
                    ten_copy.add_edge((i+1)+(len(orig_nodes)*j),(i+1)+(len(orig_nodes)*(j+1)), upper =  G._node[i]['sup_dem'] , lower = 0)
        elif G._node[i]['sup_dem'] <  0:
            for j in range(start_time_int,curr_T-1):
                if i not in removed_nodes_mat[j+1]:
                    ten_copy.add_edge((i+1)+(len(orig_nodes)*j),(i+1)+(len(orig_nodes)*(j+1)), upper =  -G._node[i]['sup_dem'] , lower = 0 )

    new_nodes = np.array(range((len(orig_nodes)*prev_T)+1, (len(orig_nodes)*curr_T)+1))

    ###in the case of a multidigraph
    if G.is_multigraph():
        for (i,j,k) in orig_edges:
            time_int_end = math.ceil((G[i][j][k]['travel_time']/60)/time_int_size)
            for m in range(0,curr_T-1):
                node_num = (i+1)+(len(orig_nodes)*m)
                if (time_int_end+m < curr_T) and (j not in removed_nodes_mat[time_int_end+m]) and (i not in removed_nodes_mat[m]):
                    ### deteremine the capacipty the edge can have based on dist to fire
                    if (i,j,k) in edge_dist_mat[m]:
                        if (i,j,k) in edge_dist_mat[m]:
                            if edge_dist_mat[m][(i,j,k)]/time_int_end < lb_percent:
                                percent_cap = 0
                            elif edge_dist_mat[m][(i,j,k)] >= time_int_end:
                                percent_cap = 1
                            else:
                                percent_cap = edge_dist_mat[m][(i,j,k)]/time_int_end
                        else:
                            percent_cap = 1
                    
                    if prev_T == 0 or (node_num)+(j-i)+(len(orig_nodes)*time_int_end) in new_nodes:
                        if (node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end)) in ten_copy.edges:
                            ten_copy[node_num][(node_num)+(j-i)+(len(orig_nodes)*time_int_end)]['upper'] += math.floor(G[i][j][k]['upper']*percent_cap)
                        else:
                            ten_copy.add_edge(node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end),upper = math.floor(G[i][j][k]['upper']*percent_cap), lower = 0)


    ##in the case of a digraph that is not a multdigraph
    elif G.is_directed():
        for (i,j) in orig_edges:
            time_int_end = math.ceil((G[i][j]['travel_time']/60)/time_int_size)
            for m in range(0,curr_T-1):
                node_num = (i+1)+(len(orig_nodes)*m)
                if (time_int_end+m < curr_T) and (j not in removed_nodes_mat[time_int_end+m]) and (i not in removed_nodes_mat[m]):                    
                    ### deteremine the capacipty the edge can have based on dist to fire
                    if (i,j) in edge_dist_mat[m]:
                        if edge_dist_mat[m][(i,j,k)]/time_int_end < lb_percent:
                            percent_cap = 0
                        elif edge_dist_mat[m][(i,j)] >= time_int_end:
                            percent_cap = 1
                        else:
                            percent_cap = edge_dist_mat[m][(i,j)]/time_int_end
                    else:
                        percent_cap = 1
                    if prev_T == 0 or (node_num)+(j-i)+(len(orig_nodes)*time_int_end) in new_nodes:
                        if (node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end)) in ten_copy.edges:
                            ten_copy[node_num][(node_num)+(j-i)+(len(orig_nodes)*time_int_end)]['upper'] += math.floor(G[i][j]['upper']*percent_cap)
                        else:
                            ten_copy.add_edge(node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end),upper = math.floor(G[i][j]['upper']*percent_cap), lower = 0)
    else:
        raise Exception("Must Input a DiGraph or MultiDiGraph")
    
    end = time.time()
    if output:
        print(f'Time to build WTEN for T = {curr_T-1}: {end-start} seconds')

    return ten_copy

#### Functions Used in Deteremining the Time Horizon for Inital Evaucation Plan
def add_s_t(G,ten, T,**kwargs):
    '''
    Inputs:
    G: NetworkX MultiDigraph of orignal road network
    ten: Networkx DiGraph of time-expanded network

    optional inputs:
    removed_nodes_mat: list of lists of nodes which have been overtaken by the fire for each time instance t = 0,..,T; default is empty lists of lists
    verbose: boolean for including pring statemens; default if False

    Output:
    ten_copy: time expanded network with a super source and sink sink node added
    '''
    start = time.time()
    ten_copy = ten.copy()
    orig_nodes = list(G.nodes)
    ten_nodes = list(ten_copy.nodes)
    removed_nodes_mat = kwargs.get('removed_nodes_mat',[ [] for _ in range(T)])
    output = kwargs.get('verbose',False)

    sup_nodes = [x for x,y in ten_copy.nodes(data=True) if y['sup_dem'] > 0]
    dem_nodes = [x for x,y in G.nodes(data=True) if y['sup_dem'] < 0]
    
    s = 0
    t = max(ten_nodes)+1
    
    ten_copy.add_node(s,name = f'{s}-{0}',level = 0,sup_dem = 0)
    ten_copy.add_node(t,name = f'{len(orig_nodes)}-{T-1}',level = T-1, sup_dem = 0)

    for i in sup_nodes:
        ten_copy.add_edge(s, i, cost = 0, upper = ten._node[i]['sup_dem'], lower = 0)

    for i in dem_nodes:
        for k in range(T-1):
            added = False
            if i in removed_nodes_mat[k+1]:
                ten_copy.add_edge((i+1) + ((k) * len(orig_nodes)),t, upper = -G._node[i]['sup_dem'], lower = 0)
                added = True
                break;
        if added is False:
            ten_copy.add_edge((i+1) + ((T-1) * len(orig_nodes)),t, upper = -G._node[i]['sup_dem'], lower = 0)

    end = time.time()
    if output:
        print(f'Add Super Source and Sink Time: {end-start} seconds')

    
    
    return ten_copy

def color_max_flow(graph, flow_dict):
    '''
    Inputs:
    graph: NetworkX DiGraph or MultiDiGraph
    flow_dict: dictionary of flow values for the given graph

    Output:
    Returns G with the added edge attribute "color"
    '''
    edges = graph.edges

    for (i,j) in edges:
        graph[i][j]['color'] = (0,0,0)
        if i in flow_dict:
            if j in flow_dict[i]:
                if flow_dict[i][j] !=0:
                    graph[i][j]['color'] = (1,0,0)
                    
    return graph

def det_num_int(G, pop, fire_orig_radii, **kwargs):
    '''
    Inputs:
    G: NetworkX MultiDiGraph
    pop: integer, population of area being evaucated
    fire_orig_radii: list of tuples of the form (lat, long, init_radius) for the circles of the fire polygon at a time t

    optional inputs:
    T: integer, given time horizon t; default is computed as the time of the longest shortest path plus the number of edges in the path
    step_size: integer, step size of increase to time horizon T; default is 1
    verbose: boolean for including print statements; default is Flase

    Outputs:
    ints: integer representing the time horizon T required for inital evaucation plan
    flow_value: integer, value of max flow solution
    flow_dict: dictionary of the flow for edges in the time-expanded network for the max-flow solution
    colored_max_flow_ten: ten whcih contains edge attribute "color"
    full_ten: ten with fire data not included for time horizon T
    rmvd_nodes_mat: list of lists of nodes overtaken by fire for all time instances t = 0,...,T
    edge_dist_mat: lists of lists of edge distances to fire for all time instances t=0,..,T
    fire_polygon_mat: lists of lists of MultiPoolygons/Polygongs representing the fire for all time instance t=0,..,T
    '''
    T = kwargs.get('T', 0)
    step_size = kwargs.get('step_size', 1)
    time_int_len = kwargs.get('time_int_len', 1)
    output = kwargs.get('verbose', False)
    lb_percent = kwargs.get('lb_percent', 0.2)
    safe_est = kwargs.get('safe_est', True)

    dem_nodes = [x for x,y in G.nodes(data=True) if y['sup_dem'] < 0]
    sup_nodes = [x for x,y in G.nodes(data=True) if y['sup_dem'] > 0]
    # shortest_times = []
    # shortest_paths = []
    max_shortest_path_length = -float('inf')
    longest_shortest_path = None
    supply_amt = 0
    demand_amt = 0
    init_T = T
    
    ### determine if problem will have a solution
    start = time.time()
    for i in sup_nodes:
        supply_amt = supply_amt + G._node[i]['sup_dem']
    for j in dem_nodes:
        demand_amt = demand_amt - G._node[j]['sup_dem']
    
    if supply_amt != demand_amt:
        raise Exception("Supply does not equal demand. Please adjust values accordingly.")
        exit()
    elif supply_amt < pop:
        raise Exception("Population is larger than supply. Not possible to evacuate everyone.")
        exit()
    end = time.time()
    if output:
        print(f'Feasability Check Completed: {end-start} seconds') 

    ###deteremine minimum number of time intervals by finding the longest time to travel between any of the sources or sinks
    if T==0:
        start = time.time()

        # Iterate over each source
        for source in sup_nodes:
            # Compute the shortest path lengths from the source to all nodes
            shortest_paths_lengths = nx.single_source_dijkstra_path_length(G, source, weight = 'travel_time')
        
            # Iterate over each sink
            for sink in dem_nodes:
                # Check if there is a path from source to sink
                if sink in shortest_paths_lengths:
                    # Get the shortest path length to this sink
                    path_length = shortest_paths_lengths[sink]
                
                    # Update if this is the longest shortest path found so far
                    if path_length > max_shortest_path_length:
                        max_shortest_path_length = path_length
                        # Get the actual path as well
                        longest_shortest_path = nx.shortest_path(G, source=source, target=sink, weight='travel_time')

        end = time.time()
        if output:
            # print(max_shortest_path_length)
            # print(longest_shortest_path)
            print(f'Longest Shortest Path Deteremined: {end-start} seconds')

        ###deteremine how many travel times were rounded in creation of TEN
        if not safe_est:
            num_rnd_ints = 0
            for i in range(len(longest_shortest_path)-1):
                edges = G.get_edge_data(longest_shortest_path[i], longest_shortest_path[i+1])
                min_travel_time = min(edge_data['travel_time'] for edge_data in edges.values())
                if min_travel_time != math.ceil(min_travel_time):
                    num_rnd_ints +=1 
        # end = time.time() 
                

    
        if safe_est:
            init_T= math.ceil((((max_shortest_path_length/60))+len(longest_shortest_path))/time_int_len) ###time zero is when num_time_ints = 1
        else:
            init_T= math.ceil((((max_shortest_path_length/60))+num_rnd_ints)/time_int_len) ###time zero is when num_time_ints = 1
        T = init_T
    prev_T = 0

    T_0 = sum([math.ceil(data['travel_time']) for _,_,data in G.edges(data=True)])
    min_cap = min([data['upper'] for _,_,data in G.edges(data=True)])*lb_percent
    max_ints = math.ceil(pop/min_cap)

    ten = nx.DiGraph()
    prev_ten = ten.copy()
    full_ten = nx.DiGraph()
    s_t_ten = nx.DiGraph()
    flow_value = 0
    flow_dict = {}
    rmvd_nodes_mat = []
    edge_dist_mat = []
    fire_polygon_mat = []
    unsafe = False

    while flow_value < pop:

        rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat = create_fire_mat(G, fire_orig_radii,T, rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat, verbose = output)
        intersection = set(sup_nodes).intersection(set(rmvd_nodes_mat[0]))
        dem_subset = set(dem_nodes).issubset(set(rmvd_nodes_mat[-1]))
        while len(intersection) !=0 and len(sup_nodes) !=0:
            print(f"One or more source nodes has already been engulfed by the fire. These nodes will be removed from consideration.")
            intersection = set(sup_nodes).intersection(set(rmvd_nodes_mat[0]))
            for node in intersection:
                G._node[node]['sup_dem'] = 0
            rmvd_nodes_mat = []
            fire_polygon_mat = []
            sup_nodes = [x for x,y in G.nodes(data=True) if y['sup_dem'] > 0]
            rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat = create_fire_mat(G, fire_orig_radii,T, rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat, verbose = output)
            intersection = set(sup_nodes).intersection(set(rmvd_nodes_mat[0]))
        if len(intersection) !=0 and len(sup_nodes) ==0:
            print(f"All sources have been overtaken by the fire, so evacuation is not possible.")
            break;
        if dem_subset:
            unsafe = True
            print(f"All sinks have been overtaken by the fire, so not everyone can evacuate by time horizon {T}.")
            print(f"Will return solution for time horizon {T-step_size}")
            T = T-1

            ten = time_expand_with_removal_dyn(G, prev_ten, time_int_len, prev_T, T, removed_nodes_mat = rmvd_nodes_mat,edge_distance_mat = edge_dist_mat, verbose = output)
            s_t_ten = add_s_t(G, ten, T, removed_nodes_mat = rmvd_nodes_mat, verbose = output)
            flow_value, flow_dict = nx.maximum_flow(s_t_ten.copy(), 0, max(list(s_t_ten.copy().nodes)),capacity = 'upper',flow_func = shortest_augmenting_path)
            break;
        else:
            ten = time_expand_with_removal_dyn(G, prev_ten, time_int_len, prev_T, T, removed_nodes_mat = rmvd_nodes_mat,edge_distance_mat = edge_dist_mat, verbose = output)
            s_t_ten = add_s_t(G, ten, T, removed_nodes_mat = rmvd_nodes_mat, verbose = output)
            start = time.time()
            flow_value, flow_dict = nx.maximum_flow(s_t_ten.copy(), 0, max(list(s_t_ten.copy().nodes)),capacity = 'upper',flow_func = shortest_augmenting_path)
            end = time.time()
            if output:
                print(f'Time for Max Flow Algorithm: {end-start} seconds')
                print('----------------------------------------------------------------------')

            if T >= min(T_0*max_ints,init_T+90):
                print('Maximum Time Horizon or Computational Limit Reached')
                break;

            prev_ten = ten.copy()
            prev_T = T
            T +=step_size
            
    if unsafe:
        ints = T
    else:
        ints = T - step_size
    if output:
        print('Construct Colored Time Expanded Network')
    full_ten_temp = time_expand_with_removal_dyn(G, nx.DiGraph(), time_int_len, 0, ints)
    full_ten = add_s_t(G,full_ten_temp, ints-step_size)
    start = time.time()
    colored_max_flow_ten = color_max_flow(s_t_ten,flow_dict)
    end = time.time()
    if output:
        print(f'Time to color edges with Flow in WTEN: {end-start} seconds')
    
    return (ints, flow_value, flow_dict, colored_max_flow_ten, full_ten, rmvd_nodes_mat, edge_dist_mat, fire_polygon_mat, T_0*max_ints) 

#### Functions for Updating the Initial Evaucation Plan

def update_ten(full_ten, G, removed_nodes_mat, edge_dist_mat, time_int_update, T, time_int_size, flow_dict, **kwargs):
    '''
    Inputs:
    full_ten: Networkx DiGraph for time-expanded network with no fire data used for construction
    G: Networkx MultiDiGraph of orignal road network
    removed_nodes_mat: lists of lists of nodes which have ben overtaken by fire for every time instance t=0,..T
    edge_dist_mat: list of lists of distance of all edges to fire for every time isntance t=0,...T
    time_int_update: integer, time instance when evaucation crews will be able to implement updated evaucation plan
    T: time horizon
    time_int_size: amount of time each time instance represents (ex. a value of 2 means there are 2 minutes between each time instance)
    flow_dict: dictionary of flow for edges in time-expanded network with fire data used in construction


    Output:
    part_ten: a truncated version of full_ten based on when what time instance the inital plan is updated
    '''

    percent_cap = 1
    output = kwargs.get('verbose', False)
    lb_percent = kwargs.get('lb_percent', 0.2)

    start = time.time()

    full_ten_copy = full_ten.copy()
    orig_nodes = list(G.nodes)
    orig_edges = list(G.edges)
    old_start_nodes = np.array(range((len(orig_nodes)*time_int_update)+1))
    t = max(list(full_ten_copy.nodes))

    for i in flow_dict:
        for j in flow_dict[i]:
            if (i in old_start_nodes) and (j not in old_start_nodes) and (flow_dict[i][j] > 0):
                full_ten_copy._node[j]['sup_dem'] += flow_dict[i][j]
        
    full_ten_copy.remove_nodes_from(old_start_nodes)
    full_ten_copy.remove_node(t)

    part_ten = nx.create_empty_copy(full_ten_copy, with_data = True)

    for i in G.nodes:
        if G._node[i]['sup_dem'] >  0:
            for j in range(time_int_update,T-1):
                if i not in removed_nodes_mat[j+1]:
                    part_ten.add_edge((i+1)+(len(orig_nodes)*j),(i+1)+(len(orig_nodes)*(j+1)), upper =  G._node[i]['sup_dem'] , lower = 0)
        elif G._node[i]['sup_dem'] <  0:
            for j in range(time_int_update,T-1):
                if i not in removed_nodes_mat[j+1]:
                    part_ten.add_edge((i+1)+(len(orig_nodes)*j),(i+1)+(len(orig_nodes)*(j+1)), upper =  -G._node[i]['sup_dem'] , lower = 0 )

    ##in the case of a multidigraph
    if G.is_multigraph():
        for (i,j,k) in orig_edges:
            time_int_end = math.ceil((G[i][j][k]['travel_time']/60)/time_int_size)
            for m in range(time_int_update,T-1):
                node_num = (i+1)+(len(orig_nodes)*m)
                if (time_int_end+m < T) and (j not in removed_nodes_mat[time_int_end+m]) and (i not in removed_nodes_mat[m]):
                    ### deteremine the capacipty the edge can have based on dist to fire
                    if (i,j,k) in edge_dist_mat[m]:
                        if edge_dist_mat[m][(i,j,k)]/time_int_end < lb_percent:
                            percent_cap = 0
                        elif edge_dist_mat[m][(i,j,k)] >= time_int_end:
                            percent_cap = 1
                        else:
                            percent_cap = edge_dist_mat[m][(i,j,k)]/time_int_end
                    else:
                        percent_cap = 1
                    if (node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end)) in part_ten.edges:
                        part_ten[node_num][(node_num)+(j-i)+(len(orig_nodes)*time_int_end)]['upper'] += math.floor(G[i][j][k]['upper']*percent_cap)
                    else:
                        part_ten.add_edge(node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end),upper = math.floor(G[i][j][k]['upper']*percent_cap), lower = 0)


    ##in the case of a digraph that is not a multdigraph
    elif G.is_directed():
        for (i,j) in orig_edges:
            # print(f'({i},{j},{k})')
            time_int_end = math.ceil((G[i][j]['travel_time']/60)/time_int_size)
            for m in range(time_int_update,T-1):
                node_num = (i+1)+(len(orig_nodes)*m)
                if (time_int_end+m < T) and (j not in removed_nodes_mat[time_int_end+m]) and (i not in removed_nodes_mat[m]):                    
                    ### deteremine the capacipty the edge can have based on dist to fire
                    if (i,j) in edge_dist_mat[m]:
                        if (i,j) in edge_dist_mat[m]:
                            if edge_dist_mat[m][(i,j)]/time_int_end < lb_percent:
                                percent_cap = 0
                            elif edge_dist_mat[m][(i,j)] >= time_int_end:
                                percent_cap = 1
                            else:
                                percent_cap = edge_dist_mat[m][(i,j)]/time_int_end
                        else:
                            percent_cap = 1

                    if (node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end)) in part_ten.edges:
                        part_ten[node_num][(node_num)+(j-i)+(len(orig_nodes)*time_int_end)]['upper'] += math.floor(G[i][j]['upper']*percent_cap)
                    else:
                        part_ten.add_edge(node_num,(node_num)+(j-i)+(len(orig_nodes)*time_int_end),upper = math.floor(G[i][j]['upper']*percent_cap), lower = 0)


    end = time.time()
    if output:
        print(f'Time to create truncated WTEN for T = {T-1}: {end -start} seconds')
    return part_ten

def merg_tens(full_ten_s_t, part_ten_s_t):
    '''
    Inputs:
    full_ten_s_t: NetworkX DiGraph of time-expanded network where fire data was not used in construction which had a super source and sink added
    part_ten_s_t: NetworkX DiGraph of the truncated time-expanded network for the updated evacuation plan

    Output:
    full_ten_copy: NetworkX DiGraph which is a combination of full_ten_s_t and part_ten_s_t
    '''
    full_ten_copy = full_ten_s_t.copy()
    part_ten_copy = part_ten_s_t.copy()
    
    t = max(list(full_ten_copy.nodes))
    full_ten_copy.remove_edges_from(full_ten_copy.edges(t))
    part_ten_copy.remove_node(0)

    full_ten_copy.remove_edges_from(list(part_ten_copy.edges))
    full_ten_copy.add_edges_from(part_ten_copy.edges(data = True))
    
    return full_ten_copy

def merge_flow_dicts(dict1, dict2):
    '''
    Inputs:
    dict1: flow dictionary for the original evaucation plan
    dict2: flow dictionart for the truncated time-expanded network used for updating the evcuation plan

    Output:
    merged_dict: dictionary of merged flow dictionaries for updated evaucation plan over entire time-expanded network
    '''
    merged_dict = dict1.copy()  # Create a copy of the first dictionary

    dict_set_1 = set(dict1)
    dict_set_2 = set(dict2)
    updated_nodes = dict_set_1.intersection(dict_set_2)

    # Set values in dict1 to zero if they're not present in dict2
    for node1 in list(updated_nodes):
        edges = dict1[node1]
        for node2, flow in edges.items():
            if node1 in dict2.keys() and node2 in dict2[node1].keys():
                if dict2[node1][node2] is not None:
                    merged_dict[node1][node2] = dict2[node1][node2]
            else:
                merged_dict[node1][node2] = 0
    
    return merged_dict

def evac_update(full_ten_s_t, G, orig_removed_nodes_mat, orig_edge_dist_mat, orig_fire_poly_mat, time_int_update, fire_time_int, T, 
                fire_orig_radii, flow_dict,**kwargs):
    '''
    Inputs:
    full_ten_s_t: NetworkX DiGraph of time-expanded network without fire used in construction
    G: NetworkX MultiDiGraph of oringla road network
    orig_removed_nodes_mat: list of lists of nodes overtaken by fire in original evacuation plan for all time instance t=0,..,T
    orig_edge_dist_mat: list of lists of distance from all edges to fire in original evaucation plan for all time instance t=0,..,T
    orig_fire_poly_mat: list of lists of fire polygons used in original evacuation plan for all time instances t=0,...,T
    time_int_update: integer, time instance where evuacation crews can make adjustements for updated plan
    fire_time_int: integer, time isntnace where fire is going to change from that in the original evuacation plan
    T: integer, time horizon
    fire_orig_radii: list of tuples of the form (lat, long, init_radius) for the circles of the fire polygon at a time t
    flow_dict: dictionary of flow for all edges in full_ten_s_t

    optional inputs:
    time_int_size: size of time difference between each time isntance (ex. a value of two means there are 2 minutes between each time instance)

    Outputs:
    part_ten_s_t: NetworkX DiGraph of truncated time-expanded network for updated evacuation plan with super source and sink
    comb_rmvd_nodes_mat: list of lists of nodes overtaken by fire for udpated evacuation plan for all time instances t=0,..,T
    comb_edge_dist_mat: list of lists of distance of all edges to fire for udpated evacuation plan for all time instances t=0,..,T
    comb_fire_poly_mat: list of lists of fire polygons used in udpated evacuation plan for all time instances t=0,..,T
    flow_dict_part: dictionary of flow for all edges int part_ten_s_t
    flow_value_part: total flow for max-flow solution on part_ten_s_t
    '''

    time_int_size = kwargs.get('time_int_size', 1)
    step_size = kwargs.get('step_size', 1)
    output = kwargs.get('verbose',False)
    pop = kwargs.get('pop', 1)
    max_T = kwargs.get('max-T', T+1)

    start = time.time()

    full_ten_copy = full_ten_s_t.copy()

    comb_rmvd_nodes_mat = []
    comb_edge_dist_mat = []
    comb_fire_poly_mat = []
    
    ### get new predicted fire data
    new_rmvd_nodes_mat, new_edge_dist_mat, new_fire_poly_mat = create_fire_mat(G, fire_orig_radii, T, [], [], [], 
                                                                              start_time_int = fire_time_int, verbose = output)
    
    ### update fire data at correct spots in original array
    comb_rmvd_nodes_mat = orig_removed_nodes_mat[:fire_time_int] + new_rmvd_nodes_mat
    comb_edge_dist_mat = orig_edge_dist_mat[:fire_time_int] + new_edge_dist_mat
    comb_fire_poly_mat = orig_fire_poly_mat[:fire_time_int] + new_fire_poly_mat
    
    ### recreate just ten that changes
    part_ten = update_ten(full_ten_copy, G, comb_rmvd_nodes_mat, comb_edge_dist_mat, time_int_update, T, time_int_size, flow_dict, verbose = output)

    ### add super source and sink to partial ten
    part_ten_s_t = add_s_t(G, part_ten, T, verbose = output)
    ### solve max flow on partial network
    start_max = time.time()
    flow_value_part, flow_dict_part = nx.maximum_flow(part_ten_s_t.copy(), 0, max(list(part_ten_s_t.copy().nodes)),capacity = 'upper',
                                                flow_func = shortest_augmenting_path)
    end_max = time.time()

    if flow_value_part < pop:

        prev_ten = part_ten.copy()
        rmvd_nodes_mat = comb_rmvd_nodes_mat
        edge_dist_mat = comb_edge_dist_mat
        fire_polygon_mat = comb_fire_poly_mat

        num_time_ints = T+step_size
        prev_num_time_ints = T
    
        while num_time_ints <= min([max_T,90]):
            dem_subset = set(dem_nodes).issubset(set(comb_rmvd_nodes_mat[-1]))
            if dem_subset:
                unsafe = True
                print(f"All sinks have been taken by the fire, not everyone can evacuate by time horizon {num_time_ints}.")
                print(f"Will return solution for time horizon {num_time_ints-1}")
                num_time_ints = num_time_ints-1

                ten = time_expand_with_removal_dyn(G, prev_ten, time_int_len, prev_num_time_ints,num_time_ints, removed_nodes_mat = rmvd_nodes_mat,
                                                   edge_distance_mat = edge_dist_mat)
                s_t_ten = add_s_t(G, ten, num_time_ints, removed_nodes_mat = rmvd_nodes_mat)
                flow_value, flow_dict = nx.maximum_flow(s_t_ten.copy(), 0, max(list(s_t_ten.copy().nodes)),capacity = 'upper',
                                                        flow_func = shortest_augmenting_path)
                break;
        

            else:
                ten = time_expand_with_removal_dyn(G, prev_ten, time_int_len, prev_num_time_ints, num_time_ints, removed_nodes_mat = rmvd_nodes_mat,
                                                   edge_distance_mat = edge_dist_mat)
                s_t_ten = add_s_t(G, ten, num_time_ints, removed_nodes_mat = rmvd_nodes_mat)
                flow_value, flow_dict = nx.maximum_flow(s_t_ten.copy(), 0, max(list(s_t_ten.copy().nodes)),capacity = 'upper',
                                                        flow_func = shortest_augmenting_path)

                prev_ten = ten.copy()
                prev_num_time_ints = num_time_ints
                num_time_ints +=step_size
                rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat = create_fire_mat(G, fire_origins_radii[-1],num_time_ints, 
                                                                                rmvd_nodes_mat,edge_dist_mat,fire_polygon_mat)

    end = time.time()

    if output:
        print(f'Time for Max Flow Algorithm: {end_max-start_max} seconds')
        print(f'Time to update evaucation plan: {end-start} seconds')
        print('Evacuation Plan Successfully Updated')
    
    return part_ten_s_t, comb_rmvd_nodes_mat, comb_edge_dist_mat, comb_fire_poly_mat, flow_dict_part, flow_value_part, T

#### Function to Covert from Time-Expanded Network Notation to Original Graph Notation
def flow_at_time_int(ten, flow_edges, G, end_time_int, **kwargs):
    '''
    Inputs:
    ten: NetworkX DiGraph for a time-expanded network
    flow_edges: list of edges which have flow in max-flow solution for given ten
    G: NetworkX MultiDiGraph for original raod network
    end_time_int: integer. time instance we plot to for flow in network

    optional input:
    start_time_int: integer. time instance we start to plot for flow in network; default is 1
    interval: boolean, switch to plot flow for an interval of time instances; default is False

    Outputs:
    Return a list of edges in the original graph notation which had flow which ends at or crosses between the time periods start_time_int and end_time_int
    '''
    start_time_int = kwargs.get('start_time_int',1) 
    interval = kwargs.get('interval',False)
    orig_edges = []
    start_node = (len(G.nodes)*(start_time_int-1))+1
    end_node = (len(G.nodes)*end_time_int)
    start_node_range = list(range(start_node,end_node+1))

    if interval:
        end_node_range = list(range(start_node,max(list(ten.nodes))))
    else:
        end_node_range = list(range((len(G.nodes)*(end_time_int-1)),max(list(ten.nodes))))
    
    for edge in flow_edges:
        if edge[0] in start_node_range and edge[1] in end_node_range:
            orig_edge = []
            for node in edge:
                nodes = [y['name'] for x,y in ten.nodes(data=True) if x == node]
                node_str = nodes[0].split('-')
                orig = int(node_str[0])-1
                orig_edge.append(orig)
            orig_edge = tuple(orig_edge)
            if orig_edge in G.edges:
                orig_edges.append(orig_edge)
            elif (orig_edge[1],orig_edge[0]) in G.edges:
                orig_edges.append((orig_edge[1],orig_edge[0]))
    return orig_edges