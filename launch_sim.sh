#!/bin/bash

# Kill existing tmux session
tmux kill-session -t mav_sim 2>/dev/null

# Kill any orphaned Gazebo/ROS processes
pkill -9 -f "ign gazebo"
pkill -9 -f "gz_bridge"
pkill -9 -f "image_bridge"
pkill -9 -f "odom_to_tf"
pkill -9 -f "robot_state_publisher"
pkill -9 ruby

# Wait for processes to die
sleep 1

# Verify clean state
if pgrep -f "ign gazebo" > /dev/null; then
    echo "WARNING: Gazebo processes still running!"
    ps aux | grep -E "ign gazebo|gz_bridge" | grep -v grep
    exit 1
fi

# Create new tmux session and launch simulation
tmux new-session -d -s mav_sim -n simulation

# Send commands to the tmux session
tmux send-keys -t mav_sim:simulation "cd ~/mav_sim" C-m
tmux send-keys -t mav_sim:simulation "source install/setup.bash" C-m
tmux send-keys -t mav_sim:simulation "ros2 launch mav_bringup mav_init.launch.py" C-m

echo "Simulation launched in tmux session 'mav_sim'"
echo "Attach with: tmux attach -t mav_sim"
echo "Detach with: Ctrl+b then d"
echo "Kill with: tmux kill-session -t mav_sim"