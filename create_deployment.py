import tarfile
import os

# Create tar.gz with proper Unix paths
with tarfile.open('deploy.tar.gz', 'w:gz') as tar:
    # Add directories
    for folder in ['agents', 'config', 'core', 'ui', 'sessions']:
        if os.path.exists(folder):
            tar.add(folder, arcname=folder)
            print(f"Added {folder}/")
    
    # Add files
    for file in ['requirements.txt', '.env', 'serper_validation_integration.py']:
        if os.path.exists(file):
            tar.add(file, arcname=file)
            print(f"Added {file}")

print("Created deploy.tar.gz with Unix paths")
