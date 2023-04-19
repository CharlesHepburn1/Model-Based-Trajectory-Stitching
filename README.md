# Model-based Trajectory Stitching
** note: soon we are updating this code to compute stitching events in parallel **

Model-based trajectory stitching is an iterative data improvement strategy which can be applied to historical datasets containing
demonstrations of sequential decisions taken to solve a complex task.

![TSFig](https://user-images.githubusercontent.com/72391441/220144478-f0a2750c-a69c-4b25-bea0-1e65a57590ac.png)


To recreate the datasets in the paper run the following line (using HalfCheetah Medium Expert as an example)
```
python3 TrajectoryStitching.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp" --reward_function "WGAN"
```
To recreate the results used in the paper run `BC_loop.py` which creates a policy by behavioural cloning for each D4RL dataset.

Note that before running `TrajectoryStitching.py`, a forward model, inverse model and reward function are pre-trained. 

### bibtex
```
@article{hepburn2022model,
  title={Model-based trajectory stitching for improved behavioural cloning and its applications},
  author={Hepburn, Charles A and Montana, Giovanni},
  journal={arXiv preprint arXiv:2212.04280},
  year={2022}
}
```
