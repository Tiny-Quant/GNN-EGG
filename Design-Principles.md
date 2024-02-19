# File Structure

```
GNN-EGG/ (repo)
  |
  |-- data/ 
  |  |-- slides/ (raw or transformed data created directly from slide data)
  |  |
  |  |-- explainee/ 
  |    |
  |    |-- model_name/
  |      |-- state_dict or checkpoint (.pt)
  |
  |-- notebooks/ (exploratory)
  |
  |-- egg_models/
  |  |
  |  |--generic_layer.py 
  |  | 
  |  |--losses.py 
  |  | 
  |  |--trainer.py 
  |  | 
  |  |-- egg_x.py (final models + extended training logic + logging)
  | 
  |-- utils/ 
  |  |-- ceograph.py (ceograph data structs + processing helper functions).
  |  | 
  |  |--segmentation_function.py 
  |  | 
  |  |--visuals.py 
  |
  |-- scripts/ (.py)
  |  |
  |  |-- train_{model name}.py (script - handles all model parameters)
  |
  |-- experiments/
  |  |
  |  |-- {model name}/ 
  |  |  |-- {experiment name}/ 
  |  |  |  |-- checkpoints/ 
  |  |  |  |
  |  |  |  |-- tensorboard/
  |  |  |  |
  |  |  |  |-- config.json (handles all training parameter + file paths)
  |  | 
  |-- archive/
```