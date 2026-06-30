#!/bin/sh
# NexusGraph Grafana init script
# Fix PostgreSQL plugin symlink (dist/ instead of source dir)
ln -snf /usr/share/grafana/public/app/plugins/datasource/grafana-postgresql-datasource/dist /usr/share/grafana/public/plugins/grafana-postgresql-datasource
# Start Grafana
exec /run.sh
