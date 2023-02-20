# Model-based Trajectory Stitching

Model-based trajectory stitching is an iterative data improvement strategy which can be applied to historical datasets containing
demonstrations of sequential decisions taken to solve a complex task.


To recreate the results in the paper run the following line (using HalfCheetah Medium Expert as an example)
```
python3 TrajectoryStitching.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp" --reward_function "WGAN"
```

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
