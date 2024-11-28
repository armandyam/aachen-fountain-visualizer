
# Optimized Route Visualization and KML Exporter for fountains in Aachen

The Python project solves the Traveling Salesman Problem (TSP) for a set of locations and provides a visualization of the optimized route on an interactive map using Folium. The application also generates a KML file for importing the optimized route into Google My Maps. The main idea was to run the TSP problem on the list of fountains in the city of Aachen in Germany. The list of these fountains can be found in the CSV file `adjusted_fountains_in_aachen.csv`, which has been parsed from the [Wiki Commons](https://commons.wikimedia.org/wiki/Fountains_in_Aachen) page. 

## Features

- **Route Optimization**: Solves the TSP using [Frigidum](https://pypi.org/project/frigidum/)'s simulated annealing and ORS API for real-world walking distances.
- **Interactive Map**: Generates an HTML map (viewable in a browser) with dropdown options for "crow flies" and "walking distance" routes.
- **KML Export**: Outputs a KML file for use in Google My Maps.
- **Duplicate Handling**: Adjusts coordinates slightly for duplicate locations to ensure valid routing.

---

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/armandyam/aachen-fountain-visualizer.git
   cd aachen-fountain-visualizer
   ```

2. **Setting up a Virtual Environment**

   It is recommended to use a virtual environment for managing dependencies. Follow these steps:

   1. **Create a virtual environment**:
      ```bash
      python -m venv venv
      ```

   2. **Activate the virtual environment**:
      - On Linux/Mac:
        ```bash
        source venv/bin/activate
        ```
      - On Windows:
        ```bash
        venv\Scripts\activate
        ```

   3. **Install required dependencies**:
      ```bash
      pip install -r requirements.txt
      ```

   4. **Deactivate the virtual environment when done**:
      ```bash
      deactivate
      ```

3. Set up the OpenRouteService API key:
   - Sign up at [OpenRouteService](https://openrouteservice.org/) to get an API key.
   - Export the key as an environment variable:
     ```bash
     export ORS_API_KEY=your_api_key
     ```

---

## Usage

### Input Format

The script uses a CSV file (`adjusted_fountains_in_aachen.csv`) as input. The file must contain the following columns:

| name             | latitude | longitude | image_url                   |
|-------------------|----------|-----------|-----------------------------|
| Fountain A        | 50.7761  | 6.0836    | https://example.com/image1.jpg |
| Fountain B        | 50.7765  | 6.0840    | https://example.com/image2.jpg |

You can replace the sample data with your own. Ensure the file is named `adjusted_fountains_in_aachen.csv` or update the script accordingly.

### Running the Script

1. Execute the script:
   ```bash
   python main.py
   ```

2. Outputs:
   - **HTML Map**: `optimized_route_with_dropdown.html` (view in any browser).
   - **KML File**: `optimized_route.kml` (import into Google My Maps).

### Example Outputs

- **HTML Map**:
  - Interactive map with dropdown to toggle between "crow flies" and "walking distance" routes.
  - Includes markers and route paths.

- **KML File**:
  - Ready to be uploaded to [Google My Maps](https://www.google.com/mymaps) for further use.

---

## Notes

- The walking distances are calculated using the OpenRouteService API. Ensure your API key is valid and has sufficient quota.
- Duplicate locations are handled automatically by adding a small random offset to the coordinates.
- The `route_cache.json` and `distance_cache.json` files are written to ensure we dont call the API for every run and also to save the optimized route for future use.

---

## Requirements

- Python 3.8+
- Install dependencies with `pip install -r requirements.txt`.

---

Enjoy exploring optimized routes for your locations! 🎉

⚠️ **Warning**: Ensure that sensitive data, such as API keys and personal location data, are not shared or committed publicly.