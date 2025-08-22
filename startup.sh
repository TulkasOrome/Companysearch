echo "Starting application setup..."

# Extract the archive if it exists
if [ -f /home/site/wwwroot/output.tar.gz ]; then
    echo "Extracting output.tar.gz..."
    cd /home/site/wwwroot
    tar -xzf output.tar.gz
    echo "Extraction complete"
else
    echo "No output.tar.gz found"
fi

# List contents to verify
echo "Current directory contents:"
ls -la /home/site/wwwroot/

# Check if ui directory exists
if [ -d /home/site/wwwroot/ui ]; then
    echo "Found ui directory"
    cd /home/site/wwwroot
    python -m streamlit run ui/streamlit_app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true
else
    echo "UI directory not found. Looking for streamlit_app.py..."
    find /home/site/wwwroot -name "streamlit_app.py" -type f
    # Try running from current directory
    cd /home/site/wwwroot
    python -m streamlit run streamlit_app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true
fi