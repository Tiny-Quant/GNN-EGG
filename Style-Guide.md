# File Structure

```
GNN-EGG/ (repo)
  |
  |-- data/ 
  |  |-- slides/ (raw or transformed data created directly from slide data)
  |  |
  |  |-- trained/ 
  |    |
  |    |-- model_name/
  |      |-- state_dict or checkpoint (.pt)
  |      |
  |      |-- training_logs (losses, grad checks, timing) (.pkl)
  |
  |-- notebooks/ (exploratory)
  |
  |-- egg_models/
  |  |
  |  |-- egg_components/ (layers, losses, trainer)
  |  | 
  |  |-- egg_x.py (final model built from components)
  |  |
  |  |-- train_egg_x.py (runnable script, terminal options, logging)
  | 
  |-- scripts/ (.py)
  |     |-- template/ (includes files for unit tests)
  |     |
  |     |-- utils/ (mostly data processing or helpers)
  |     |
  |     |-- ceograph/
  |     | 
  |     |-- visuals/
  |     |
  |
  |-- archive/
```