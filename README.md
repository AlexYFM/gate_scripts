# File Structure
The files in this repo should all be put in the `mav_sim` directory given that you follow the steps in the [Mav Sim repo](https://github.com/UIUC-Robotics/mav_simulator/).

# Data Collection
You should be able to collect data by running `python mav_sim\point_label_rand_real_train.py` while the simulator is running, headless or otherwise. The data collection script currently just teleports to random positions around a gate. You should see the images and labels created in the `mav_sim\dataset` directory.

## Note on Drone Model
I modified the drone's model.sdf file to turn off gravity so that my teleport function would work properly; you'll have to do this as well if you want to make my script work. Otherwise, you can modify the data collection script to collect data by just taking a certain path instead, which obviously doesn't require teleporting to work. 

# Training
You should be able to train the pretrained yolo model using the test_train script. Comment/uncomment out lines to either debug or run a more realistic training function. You should be able to view intermediate results in the `mav_sim\runs` directory, where you can verify that the collection script is labeling correct and the model is predicting more or less correctly.

## Dataset Formatting
If you take a look at the dataset.yaml, you will notice that there are currently two classes configured: single gate and double. This is reflected in `world_config.json`, where the double gates have two extra waypoints corresponding to the middle bar. I think it's possible to add even more gates (e.g., a triple gate or double gate horizontally-oriented, but you'd have ensure that world_config is updated, and if you add extra waypoints, ensure that the other classes have zeroed out visibility for said waypoints.
