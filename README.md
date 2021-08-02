# Stunfisk-RL: Pokemon reinforcement learning

This is a simple little project to investigate how well RL can do on pokemon, relying mostly on the great work of the poke-env library. I've applied it to stable-baselines3, allowing easy testing of a variety of RL algorithms.

Check out my blog post on this [here](https://ivison.id.au/2021/08/02/pokerl/).

## Setup

`pip install -r requirements.txt` should install all you need.

You'll need a pokemon showdown server running to use this. Clone showdown:
`git clone https://github.com/smogon/pokemon-showdown.git`

Then run this command in the background in the pokemon showdown folder:
`node pokemon-showdown start --no-security`

You might have to do some setup (e.g. running `npm i` or `./build`).

## Running it yourself

To train yourself, with the pokemon showdown server above running, just run `experiment.py`. By default, this will train and evaluate on a full randoms setup, but take a look at the code to see ways to train with set teams or similar.

## Demo Setup

To run the gradio demo discussed in my blog post, run `move_predict_api.py`. You don't need showdown running in the backgronund for this. It'll download a model (which takes some time), so make sure you have internet connection.
