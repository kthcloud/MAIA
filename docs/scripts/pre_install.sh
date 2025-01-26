cp README.md docs/source/README.md
cp MAIA.png docs/source/MAIA.png
cp Workspace.png docs/source/Workspace.png
mkdir -p docs/source/dashboard/image/README
cp dashboard/image/README/* docs/source/dashboard/image/README/
cp -r docker/MAIA-Workspace/Tutorials docs/source/apidocs/tutorials/MAIA-Workspace
python docs/scripts/generate_tutorials_rst.py