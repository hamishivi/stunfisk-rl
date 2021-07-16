#!/bin/bash
# required items
apt -y install sudo vim wget curl git
# install node 14
sudo apt update
curl -sL https://deb.nodesource.com/setup_14.x | sudo bash -
sudo apt -y install nodejs vim
# git stuff
git clone https://github.com/smogon/pokemon-showdown.git
git clone https://hamishivi:0baa31c91fc58c61cb8d0163d9d252ddfe09f4c3@github.com/hamishivi/stunfisk-rl.git

# build showdown
cd pokemon-showdown
./build
cd ..

# install python deps
pip install wandb poke-env git+https://github.com/DLR-RM/stable-baselines3 yacs tensorboard

echo "Run node pokemon-showdown start --no-security in a thing"
echo "And then you're good to go!"
