docker run -p 18888:8888 --rm -v /mnt/workspaces/DynamicTopography:/workspace/Data siwill22/pygplates-ubuntu14

docker pull siwill22/pygplates-ubuntu14

1 Generate plate frame grids
http://130.56.249.211:18888/notebooks/Data/new_dir/pygplates-citcom/dynamic-topography/Plate-Frame-Dynamic-Topography.ipynb

2 Generate GMT images:
http://130.56.249.211:18888/notebooks/Data/new_dir/pygplates-citcom/dynamic-topography/Dynamic-Topography-GMT.ipynb

3. Reduce the size of jpg:
for f in *.jpg; do convert "$f" -strip -resize 1600x800 -quality 90 "$f"; done

4 Rename files
http://130.56.249.211:8887/notebooks/dynamic_topography_file_rename%20.ipynb


Generate delta grids
http://130.56.249.211:18888/notebooks/Data/new_dir/pygplates-citcom/dynamic-topography/Plate-Frame-Dynamic-Topography.ipynb

Create CPT file
os.system('gmt makecpt -Cpolar -T-50/50/5 -D > %s' % cptfile)

Make sure to use the correct rotation model

Reduce the size of grids(not working for dynamic topography profile)
find .  -iname "*.nc" | xargs -l -i gdal_translate -of GMT -outsize 1201 601 {} ../PlateFrame/{}


