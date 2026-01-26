#!/bin/bash
# run_experiment.sh - Convenience script to run the experiment

set -e

echo "=================================="
echo "OU vs Superlinear Langevin Experiment"
echo "=================================="
echo ""

# Build the Docker image
echo "🔨 Building Docker image..."
docker compose build

# Run the experiment
echo ""
echo "🚀 Running experiment..."
docker compose run --rm experiment python main.py

echo ""
echo "✅ Experiment completed!"
echo "Check your WandB dashboard for results"