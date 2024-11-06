mkdir data
cd data
mkdir datasets
cd datasets
wget -O multinav.zip "http://aspis.cmpt.sfu.ca/projects/multion/multinav.zip"
unzip multinav.zip && rm multinav.zip
cd ../
wget -O objects.zip "http://aspis.cmpt.sfu.ca/projects/multion/objects.zip"
unzip objects.zip && rm objects.zip
wget -O default.phys_scene_config.json "http://aspis.cmpt.sfu.ca/projects/multion/default.phys_scene_config.json"
