1) To add region map in kibana add following line in kibana.yml (/etc/kibana/kibana.yml)
  - xpack.maps.showMapVisualizationTypes: true

2) goto Management -> Stack Management -> Kibana -> Saved Objects -> Import and then upload the export.ndjson file

  - This will setup all the dashboard and index settings
