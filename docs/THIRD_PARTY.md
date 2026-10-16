# Upstream dependencies

GLSD uses the response, recognition and evaluation interfaces of [ME-TST](https://github.com/zizheng-guo/ME-TST) and [BoostingVRME](https://github.com/zizheng-guo/BoostingVRME).

Configure the matching upstream checkout for each response cache. It must provide `training_utils.py` and `Utils/`, including the required mean-average-precision modules. Install the upstream project's dependencies in the environment used for that pipeline.

`requirements.txt` lists the dependencies for the GLSD decoder and synthetic example. Model training and response generation use the upstream environments.

Upstream projects retain their own licenses. The GLSD MIT license applies to original GLSD code; datasets, model checkpoints and upstream source are obtained separately.
