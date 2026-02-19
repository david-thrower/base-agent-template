#!/bin/bash
set -e

# If the vector DB doesn't exist in the mounted volume, initialize it
if [ ! -d "/data/user/chroma_db" ] || [ -z "$(ls -A /data/user/chroma_db 2>/dev/null)" ]; then
    echo "Initializing ChromaDB in mounted volume..."
    python -u 0_load_db.py
else
    echo "Using existing ChromaDB from volume..."
fi

# Now run the agent
exec python -u 2_run_agent.py
