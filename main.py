import pandas as pd
import folium
from routingpy import ORS
import time
import json
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import frigidum
from frigidum.examples import tsp
import os

# Get the API key from the environment variable
api_key = os.getenv("ORS_API_KEY")

if not api_key:
    raise ValueError("The ORS_API_KEY environment variable is not set.")

# Constants
CACHE_FILE = 'route_cache.json'
CACHE_DISTANCE_FILE = "distance_cache.json"

from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

def write_kml_file_from_cache(cache_path, kml_file):
    """
    Write a KML file using the optimized order and metadata directly from the cache file.

    Args:
        cache_path (str): Path to the JSON cache file.
        kml_file (str): Path to the output KML file.
    """
    import json

    # Load cache data
    with open(cache_path, "r") as f:
        cache = json.load(f)

    # Get walking path and order data
    walking_data = cache.get("walking_distance", {})
    path_data = walking_data.get("path", [])
    order_data = walking_data.get("order", [])
    if not path_data or not order_data:
        raise ValueError("Walking path or order data not found in the cache.")

    # Prepare KML structure
    kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, "Document")

    # Add document name
    name = SubElement(document, "name")
    name.text = "Walking Route with Optimized Order"

    # Add styles for markers
    styles = {
        "start": {
            "id": "startMarker",
            "icon_url": "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        },
        "end": {
            "id": "endMarker",
            "icon_url": "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"
        },
        "default": {
            "id": "defaultMarker",
            "icon_url": "http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png"
        }
    }

    for style_key, style_data in styles.items():
        style = SubElement(document, "Style", id=style_data["id"])
        icon_style = SubElement(style, "IconStyle")
        icon = SubElement(icon_style, "Icon")
        href = SubElement(icon, "href")
        href.text = style_data["icon_url"]

    # Add placemarks based on the order data
    placemarks_with_cdata = []
    for idx, point in enumerate(order_data):
        lat, lon = point["lat"], point["lon"]
        name = point.get("name", "Unnamed")
        image_url = point.get("image_url", "")

        placemark = SubElement(document, "Placemark")

        pname = SubElement(placemark, "name")
        pname.text = name

        # Prepare CDATA content for description
        description_content = (
            f"<img src='{image_url}' width='200' /><br>{name}"
            if image_url
            else name
        )
        placemarks_with_cdata.append((placemark, description_content))

        point_elem = SubElement(placemark, "Point")
        coordinates = SubElement(point_elem, "coordinates")
        coordinates.text = f"{lon},{lat},0"

        # Assign styles for start, end, and other markers
        style_url = SubElement(placemark, "styleUrl")
        if idx == 0:  # Start marker
            style_url.text = "#startMarker"
        elif idx == len(order_data) - 1:  # End marker
            style_url.text = "#endMarker"
        else:
            style_url.text = "#defaultMarker"

    # Add walking path as a LineString
    placemark = SubElement(document, "Placemark")

    line_name = SubElement(placemark, "name")
    line_name.text = "Walking Path"

    line_string = SubElement(placemark, "LineString")
    tessellate = SubElement(line_string, "tessellate")
    tessellate.text = "1"

    coordinates = SubElement(line_string, "coordinates")
    path_coordinates = "\n".join(
        f"{point['lon']},{point['lat']},0" for point in path_data
    )
    coordinates.text = path_coordinates

    # Serialize XML and inject CDATA descriptions
    raw_kml = tostring(kml, "utf-8").decode("utf-8")

    # Inject CDATA descriptions
    for placemark, description_content in placemarks_with_cdata:
        placemark_id = tostring(placemark, "utf-8").decode("utf-8")
        cdata_description = f"<description><![CDATA[{description_content}]]></description>"
        raw_kml = raw_kml.replace(
            placemark_id,
            placemark_id.replace("<Placemark>", f"<Placemark>{cdata_description}")
        )

    # Pretty-print the final KML
    pretty_kml = xml.dom.minidom.parseString(raw_kml).toprettyxml(indent="  ")

    # Write KML file
    with open(kml_file, "w") as f:
        f.write(pretty_kml)

    print(f"KML file written to {kml_file}")


from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points on the Earth.

    Args:
        lat1, lon1 (float): Latitude and longitude of the first point.
        lat2, lon2 (float): Latitude and longitude of the second point.

    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  # Convert degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_closest_to_centroid(coordinates):
    """
    Find the index of the fountain closest to the geometric center (centroid).

    Args:
        coordinates (list of tuple): List of (latitude, longitude) for fountains.

    Returns:
        int: Index of the fountain closest to the centroid.
    """
    # Calculate the centroid
    avg_lat = sum(lat for lat, lon in coordinates) / len(coordinates)
    avg_lon = sum(lon for lat, lon in coordinates) / len(coordinates)
    
    # Find the closest fountain
    min_distance = float("inf")
    closest_index = -1
    for i, (lat, lon) in enumerate(coordinates):
        distance = haversine(avg_lat, avg_lon, lat, lon)  # Use haversine for accurate distances
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index


def load_distance_cache():
    """
    Load the distance cache from a file.
    """
    if os.path.exists(CACHE_DISTANCE_FILE):
        with open(CACHE_DISTANCE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_distance_cache(cache):
    """
    Save the distance cache to a file.
    """
    with open(CACHE_DISTANCE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_distance_with_cache(client, start, end, profile="foot-walking"):
    """
    Get the distance and path geometry between two points, using the cache if available.

    Args:
        client (ORS): ORS API client.
        start (tuple): (latitude, longitude) of the start point.
        end (tuple): (latitude, longitude) of the end point.
        profile (str): Routing profile (default: "foot-walking").

    Returns:
        tuple: (distance in kilometers, geometry as a list of (lat, lon) points)
    """
    # Load cache
    cache = load_distance_cache()
    key = f"{start[0]},{start[1]},{end[0]},{end[1]}"
    reverse_key = f"{end[0]},{end[1]},{start[0]},{start[1]}"  # Symmetric

    # Check if the distance and geometry are already cached
    if key in cache:
        return cache[key]["distance"], cache[key]["geometry"]
    if reverse_key in cache:  # Distances are symmetric
        return cache[reverse_key]["distance"], cache[reverse_key]["geometry"][::-1]

    # If not cached, query the API
    try:
        route = client.directions(
            locations=[[start[1], start[0]], [end[1], end[0]]],
            profile=profile,
        )
        distance = route.distance / 1000  # Convert meters to kilometers
        geometry = [{"lat": coord[1], "lon": coord[0]} for coord in route.geometry]
        # Add to cache
        cache[key] = {"distance": distance, "geometry": geometry}
        save_distance_cache(cache)
        return distance, geometry
    except Exception as e:
        print(f"Error fetching distance between {start} and {end}: {e}")
        return 1e6, []  # Assign a large value and empty geometry to avoid breaking the solver


def build_walking_distance_matrix(client, coordinates):
    """
    Build the walking distance matrix and collect the geometry for the walking paths.

    Args:
        client (ORS): ORS API client.
        coordinates (list of tuple): List of (latitude, longitude) coordinates.

    Returns:
        tuple: (distance matrix, walking geometries as a dict).
    """
    n = len(coordinates)
    walking_matrix = [[0] * n for _ in range(n)]
    walking_geometries = {}  # Store geometries for each pair of points

    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle (symmetric matrix)
            if i != j:
                start = coordinates[i]
                end = coordinates[j]
                distance, geometry = get_distance_with_cache(client, start, end)
                walking_matrix[i][j] = distance
                walking_matrix[j][i] = distance  # Symmetric
                walking_geometries[(i, j)] = geometry
                walking_geometries[(j, i)] = geometry[::-1]  # Reverse geometry for reverse direction

    return walking_matrix, walking_geometries



def solve_concorde(matrix):
    """
    Solve the TSP using Concorde solver.

    Args:
        matrix (numpy.ndarray): A symmetric distance matrix.

    Returns:
        list: Optimal tour as a list of indices.
    """
    # Convert the matrix to integer as required by Concorde
    matrix = (matrix * 1e6).astype(int)  # Scale up to avoid floating-point issues
    solver = TSPSolver.from_data(
        xs=[],
        ys=[],
        matrix=matrix,
        norm="EUC_2D"  # Concorde doesn't use this when a distance matrix is provided
    )
    solution = solver.solve()
    return solution.tour


def calculate_distance_matrix(coordinates):
    """
    Calculate the pairwise distance matrix between coordinates.
    """
    n = len(coordinates)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                r = 6371  # Earth's radius in kilometers
                matrix[i][j] = c * r
    
    # Convert to numpy array
    matrix = np.array(matrix)
    
    # Debugging output
    if np.isnan(matrix).any():
        print("Error: Distance matrix contains NaN values.")
    if np.isinf(matrix).any():
        print("Error: Distance matrix contains infinite values.")
    if (matrix == 0).any() and not np.allclose(np.diag(np.diag(matrix)), matrix):
        print("Warning: Distance matrix contains unexpected zeros.")
    
    # Replace any zero off-diagonal elements with a small positive value
    matrix[matrix == 0] = 1e-6

    return matrix

import numpy as np

def rotate_tour_to_start(tour, start_index):
    """
    Rotate the TSP tour to ensure it starts at the specified starting index.

    Args:
        tour (list or numpy.ndarray): List of indices representing the TSP tour.
        start_index (int): Index of the desired starting point.

    Returns:
        list: Rotated tour starting from the specified index.
    """
    # Find the position of the start index
    start_position = np.where(tour == start_index)[0][0]
    # Rotate the tour
    return list(tour[start_position:]) + list(tour[:start_position])



def simple_tsp_solver_with_fixed_start(distance_matrix, start_index):
    """
    Solve the TSP with a fixed starting point.

    Args:
        distance_matrix (list of list of float): Distance matrix.
        start_index (int): Index of the desired starting point.

    Returns:
        list: Optimal TSP tour starting from the specified index.
    """
    # Solve the TSP using the existing solver
    tour = simple_tsp_solver(distance_matrix)

    # Rotate the tour to start at the desired point
    return rotate_tour_to_start(tour, start_index)


def simple_tsp_solver(distance_matrix):
    """
    Solve TSP using Frigidum's simulated annealing.

    Args:
        distance_matrix (list of list of float): Distance matrix.

    Returns:
        list: Optimal tour as a list of indices.
    """
    # Convert the distance matrix to a numpy array
    distance_matrix = np.array(distance_matrix)

    # Ensure the matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix must be symmetric for TSP solving.")
    
    # Debugging: Check for zeros outside the diagonal
    if (distance_matrix == 0).any() and not np.allclose(np.diag(np.diag(distance_matrix)), distance_matrix):
        print("Warning: Distance matrix contains unexpected zeros.")
    
    # Set up Frigidum's TSP environment
    tsp.dist_eu = distance_matrix
    tsp.nodes_count = len(distance_matrix)
    tsp.nodes = np.array([[i, j] for i, j in enumerate(range(len(distance_matrix)))])

    # Solve using simulated annealing
    local_opt = frigidum.sa(
        random_start=tsp.random_start,
        objective_function=tsp.objective_function,
        neighbours=[
            tsp.euclidian_bomb_and_fix,
            tsp.euclidian_nuke_and_fix,
            tsp.route_bomb_and_fix,
            tsp.route_nuke_and_fix,
            tsp.random_disconnect_vertices_and_fix,
        ],
        copy_state=frigidum.annealing.naked,
        T_start=5,
        alpha=0.5,
        T_stop=0.001,
        repeats=100,
        post_annealing=tsp.local_search_2opt,
    )
    return local_opt[0]  # Return the optimal route as a list of indices


def save_routes_to_cache(coordinates, names, image_urls):
    """
    Calculate and save routes to a JSON file for faster access.
    Solves the TSP for both "crow flies" and "walking distance."

    Args:
        coordinates (list of tuple): List of (latitude, longitude) for fountains.
        names (list of str): List of fountain names.
        image_urls (list of str): List of fountain image URLs.
        fixed_start_index (int): Index of the fixed starting point.
    """
    cache = {"crow_flies": {"path": [], "distance": 0}, "walking_distance": {"path": [], "distance": 0}}
    
        # Determine the fixed starting point
    fixed_start_index = find_closest_to_centroid(coordinates)

    # Crow Flies Path
    crow_flies_matrix = calculate_distance_matrix(coordinates)
    crow_flies_tour = simple_tsp_solver_with_fixed_start(crow_flies_matrix, fixed_start_index)
    ordered_coords_crow = [coordinates[i] for i in crow_flies_tour]
    ordered_names_crow = [names[i] for i in crow_flies_tour]
    ordered_images_crow = [image_urls[i] for i in crow_flies_tour]
    cache['crow_flies']['path'] = [
        {"lat": coord[0], "lon": coord[1], "name": name, "image_url": image_url}
        for coord, name, image_url in zip(ordered_coords_crow, ordered_names_crow, ordered_images_crow)
    ]
    cache['crow_flies']['distance'] = sum(
        crow_flies_matrix[crow_flies_tour[i]][crow_flies_tour[i+1]]
        for i in range(len(crow_flies_tour) - 1)
    )

    # Walking Distance Path
    client = ORS(api_key=api_key)
    walking_matrix, walking_geometries = build_walking_distance_matrix(client, coordinates)
    walking_tour = simple_tsp_solver_with_fixed_start(walking_matrix, fixed_start_index)
    ordered_coords_walk = [coordinates[i] for i in walking_tour]
    ordered_names_walk = [names[i] for i in walking_tour]
    ordered_images_walk = [image_urls[i] for i in walking_tour]

    # Collect the full geometry for the walking route
    full_geometry = []
    for i in range(len(walking_tour) - 1):
        full_geometry.extend(walking_geometries[(walking_tour[i], walking_tour[i+1])])

    cache['walking_distance']['path'] = [
        {"lat": coord["lat"], "lon": coord["lon"]}
        for coord in full_geometry
    ]
    cache['walking_distance']['distance'] = sum(
        walking_matrix[walking_tour[i]][walking_tour[i+1]]
        for i in range(len(walking_tour) - 1)
    )

    cache['walking_distance']['order'] = [
        {"lat": coord[0], "lon": coord[1], "name": name, "image_url": image_url}
        for coord, name, image_url in zip(ordered_coords_walk, ordered_names_walk, ordered_images_walk)
    ]

    # Save to JSON
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)



def load_routes_from_cache():
    """Load routes from the JSON cache file."""
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

def plot_routes_with_dropdown(coordinates, names):
    """Generate the map with a dropdown for route types."""
    # Load routes from the cache
    routes = load_routes_from_cache()
    
    # Initialize the map
    m = folium.Map(location=[50.776351, 6.083862], zoom_start=13)
    
    # Embed the routes JSON into the HTML
    json_data = json.dumps(routes)
    dropdown_html_js = f"""
    <script>
        var paths = {json_data}; // Embedded JSON data

        var myMap = L.map('map_id').setView([50.776351, 6.083862], 13);
        var tileLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(myMap);

        function updateMap(routeType) {{
            clearMap();

            if (paths && paths[routeType]) {{
                var route = paths[routeType].path;
                var distance = paths[routeType].distance.toFixed(2); // Format distance to 2 decimal places
                var latlngs = [];

                // Update distance label
                document.getElementById("distanceLabel").innerText = "Total Distance: " + distance + " km";

                if (routeType === "crow_flies") {{
                    // Add markers for all points in the "crow flies" route
                    for (var i = 0; i < route.length; i++) {{
                        var point = route[i];
                        latlngs.push([point.lat, point.lon]);

                        // Determine the color and tooltip content based on the marker type
                        var markerColor, tooltipLabel;
                        if (i === 0) {{
                            markerColor = 'red';
                            tooltipLabel = `Start: ${{point.name}}`;
                        }} else if (i === route.length - 1) {{
                            markerColor = 'blue';
                            tooltipLabel = `End: ${{point.name}}`;
                        }} else {{
                            markerColor = 'green';
                            tooltipLabel = `Point ${{i + 1}}: ${{point.name}}`;
                        }}

                        // Construct the tooltip HTML
                        var tooltipContent = `
                            <div style="text-align: center;">
                                <strong>${{tooltipLabel}}</strong><br>
                                <img src="${{point.image_url}}" alt="${{point.name}}" style="width:150px; height:150px; object-fit: cover; margin-top:5px;">
                            </div>
                        `;

                        L.circleMarker([point.lat, point.lon], {{
                            radius: 6,
                            color: markerColor,
                            fillColor: markerColor,
                            fillOpacity: 0.8,
                            title: point.name
                        }}).addTo(myMap)
                        .bindTooltip(tooltipContent, {{ permanent: false, direction: "top", className: "custom-tooltip" }});
                    }}
                }} else if (routeType === "walking_distance") {{
                    // Only add polyline for the "walking distance" route
                    for (var i = 0; i < route.length; i++) {{
                        latlngs.push([route[i].lat, route[i].lon]);
                    }}

                    // Add markers only at fountains from the "crow flies" route
                    for (var i = 0; i < paths["walking_distance"].order.length; i++) {{
                        var point = paths["walking_distance"].order[i];

                        // Determine the color and tooltip content based on the marker type
                        var markerColor, tooltipLabel;
                        if (i === 0) {{
                            markerColor = 'red';
                            tooltipLabel = `Start: ${{point.name}}`;
                        }} else if (i === paths["walking_distance"].order.length - 1) {{
                            markerColor = 'blue';
                            tooltipLabel = `End: ${{point.name}}`;
                        }} else {{
                            markerColor = 'green';
                            tooltipLabel = `Point ${{i + 1}}: ${{point.name}}`;
                        }}

                        // Construct the tooltip HTML
                        var tooltipContent = `
                            <div style="text-align: center;">
                                <strong>${{tooltipLabel}}</strong><br>
                                <img src="${{point.image_url}}" alt="${{point.name}}" style="width:150px; height:150px; object-fit: cover; margin-top:5px;">
                            </div>
                        `;

                        L.circleMarker([point.lat, point.lon], {{
                            radius: 6,
                            color: markerColor,
                            fillColor: markerColor,
                            fillOpacity: 0.8,
                            title: point.name
                        }}).addTo(myMap)
                        .bindTooltip(tooltipContent, {{ permanent: false, direction: "top", className: "custom-tooltip" }});
                    }}
                }}

                // Add polyline
                L.polyline(latlngs, {{
                    color: routeType === 'crow_flies' ? 'red' : 'green',
                    weight: 3,
                    opacity: 0.7
                }}).addTo(myMap);
            }}
        }}

        function clearMap() {{
            myMap.eachLayer(function(layer) {{
                if (layer !== tileLayer) {{
                    myMap.removeLayer(layer);
                }}
            }});
        }}

        // Render the default "crow flies" route immediately
        document.addEventListener("DOMContentLoaded", function() {{
            updateMap("crow_flies");
        }});
    </script>
    <div style="position: fixed; top: 10px; left: 80px; z-index: 9999; background-color: white; padding: 10px;">
        <label for="routeType">Select Route Type:</label>
        <select id="routeType" onchange="updateMap(this.value)">
            <option value="crow_flies">As the Crow Flies</option>
            <option value="walking_distance">Walking Distance</option>
        </select>
        <br>
        <label id="distanceLabel" style="margin-top: 5px; display: block;">Total Distance: 0.00 km</label>
    </div>
    <style>
        .custom-tooltip {{
            background: white;
            border: 1px solid #ccc;
            padding: 5px;
            border-radius: 5px;
            text-align: center;
        }}
        .custom-tooltip img {{
            display: block;
            margin: 0 auto;
            width: 200px;
            height: 200px;
            object-fit: cover;
        }}
    </style>
    """

    # Add HTML/JS to the map
    m.get_root().html.add_child(folium.Element(f'<div id="map_id" style="width: 100%; height: 100vh;"></div>'))
    m.get_root().html.add_child(folium.Element(dropdown_html_js))
    
    return m

def add_random_offset(lat, lon, distance):
    """
    Adds a small random offset to a coordinate.
    
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        distance (float): Distance in meters to offset.
    
    Returns:
        tuple: New latitude and longitude with the offset applied.
    """

    # Constants for the random offset
    EARTH_RADIUS = 6371000  # Radius of Earth in meters
    OFFSET_DISTANCE = 10  # Distance in meters to offset duplicate coordinates

    angle = np.random.uniform(0, 2 * np.pi)
    d_lat = (distance / EARTH_RADIUS) * (180 / np.pi)
    d_lon = d_lat / np.cos(np.radians(lat))
    
    return lat + d_lat * np.sin(angle), lon + d_lon * np.cos(angle)

def main():
    # # Read data
    df = pd.read_csv('adjusted_fountains_in_aachen.csv')
    coordinates = list(zip(df['latitude'], df['longitude']))
    seen = {}
    adjusted_coordinates = []
    # Constants for the random offset
    EARTH_RADIUS = 6371000  # Radius of Earth in meters
    OFFSET_DISTANCE = 10  # Distance in meters to offset duplicate coordinates

    for idx, (lat, lon) in enumerate(coordinates):
        if (lat, lon) in seen:
            # Add a random offset if coordinates are duplicate
            new_lat, new_lon = add_random_offset(lat, lon, OFFSET_DISTANCE)
            while (new_lat, new_lon) in seen:
                new_lat, new_lon = add_random_offset(lat, lon, OFFSET_DISTANCE)
            adjusted_coordinates.append((new_lat, new_lon))
            seen[(new_lat, new_lon)] = idx
        else:
            adjusted_coordinates.append((lat, lon))
            seen[(lat, lon)] = idx

    # Update the DataFrame with adjusted coordinates
    df['latitude'] = [coord[0] for coord in adjusted_coordinates]
    df['longitude'] = [coord[1] for coord in adjusted_coordinates]

    names = df['name'].tolist()
    image_urls = df['image_url'].tolist()
    
        # Save or process the adjusted DataFrame
    # df.to_csv('adjusted_fountains_in_aachen.csv', index=False)

    print("Adjusted coordinates for duplicate points have been saved.")

    # Save routes to cache
    # Assume the starting point is the first fountain (index 0)
    save_routes_to_cache(coordinates, names, image_urls)

    
    # Create map with dropdown
    m = plot_routes_with_dropdown(coordinates, names)
    m.save('optimized_route_with_dropdown.html')
    print("Map saved to 'optimized_route_with_dropdown.html'.")

    write_kml_file_from_cache("route_cache.json", "optimized_route.kml")



if __name__ == "__main__":
    main()
