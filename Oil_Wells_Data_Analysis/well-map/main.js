import Map from 'ol/Map.js';
import Overlay from 'ol/Overlay.js'; // Object positioning
import View from 'ol/View.js';
import VectorSource from 'ol/source/Vector.js';
import VectorLayer from 'ol/layer/Vector.js';
import TileLayer from 'ol/layer/Tile.js';
import {fromLonLat} from 'ol/proj.js';
import Feature from 'ol/Feature.js';
import CircleStyle from 'ol/style/Circle.js';
import Fill from 'ol/style/Fill.js';
import Stroke from 'ol/style/Stroke.js';
import OSM from 'ol/source/OSM.js';
import Point from 'ol/geom/Point.js';
import Papa from 'papaparse';
import Style from 'ol/style/Style.js';

// Initialize popup elements (from OpenLayer documention).
const container = document.getElementById('popup');
const content = document.getElementById('popup-content');
const closer = document.getElementById('popup-closer');

// Create overlay to anchor popups (from OpenLayer documentation).
const overlay = new Overlay({
  element: container,
  autoPan: {
    animation: {
      duration: 250,
    },
  },
});

// Close popup on click (from OpenLayer documentation)
closer.onclick = function () {
  overlay.setPosition(undefined);
  closer.blur();
  return false;
};

// Initialize base map layer.
const baseLayer = new TileLayer({
  source: new OSM(),
});

// Initialize storage and rendering of wells.
const vectorSource = new VectorSource();
const vectorLayer = new VectorLayer({
  source: vectorSource
});

// Create map
const map = new Map({
  layers: [baseLayer, vectorLayer],
  overlays: [overlay],
  target: 'map',
  view: new View({
    // Apply longitude/latitude formatting
    center: fromLonLat([-103.7, 48.1]),
    zoom: 2,
  }),
});

// Load well data
fetch(`${import.meta.env.BASE_URL}wells.csv`)
  .then(response => response.text())
  .then(csv => {
    // Parse CSV to JSON format 
    Papa.parse(csv, {
      header: true,
      skipEmptyLines: true,
      complete: function(results) {
        results.data.forEach(addWell);

        // Fit wells to a bounding box 
        map.getView().fit(vectorSource.getExtent(), {
          padding: [50,50,50,50],
          duration: 1000
        });
      }
    });
  });

// Add wells to VectorSource
function addWell(well) {
  // Extract coordinates
  const lon = parseFloat(well.de_longitude);
  const lat = parseFloat(well.de_latitude);

  if (isNaN(lon) || isNaN(lat)) return;

  // Create coordinate point
  const feature = new Feature({
    geometry: new Point(fromLonLat([lon, lat])),
    data: well
  });
 
  // Draw pushpin marker
  feature.setStyle(
    new Style({
      image: new CircleStyle({
        radius: 6,
        fill: new Fill({ color: 'red' }),
        stroke: new Stroke({ color: 'white', width: 1 })
      })
    })
  );
  
  // Add feature to fevtor source
  vectorSource.addFeature(feature);
}
 
// Handle clicked features
map.on('singleclick', function (evt) {
  // Check for a feature marker
  const feature = map.forEachFeatureAtPixel(evt.pixel, f => f);

  if (feature) {
    // Retrieve feature data
    const data = feature.get('data');
    overlay.setPosition(feature.getGeometry().getCoordinates());
    
    // Display well data
    content.innerHTML = `
      <strong>${data.well_name}</strong><br>
      Operator: ${data.de_operator}<br>
      Status: ${data.de_status}<br>
      Oil (BBLs): ${data.de_oil_bbls}<br>
      State: ${data.state}<br>
      Nearest City: ${data.de_closest_city}<br>
      Production Start Date: ${data.de_production_start}<br>
      Production End Date: ${data.de_production_end}
    `;
  } else {
    overlay.setPosition(undefined);
  }
});