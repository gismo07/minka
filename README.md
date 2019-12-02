<img src="https://ll-srv-web.livinglab.local:3000/Johann.Haselberger/MINKA/raw/branch/master/assets/Minka.png">

## About
Minka is ment to be a middleware between your optimization problem and a modular set of supporting tools to:

* automatically perform hyperparameter optimization
* store the results, parameter sets and custom data
* compare different training runs
* visualize the results and training processes

The aim of the project is to provide a simple as possible interface without the need to change your existing code.

## One line to change it all 
Could it be easier than just adding one line?
```python
minka(yourOptimizationTask, 'configuration.json').opt(numberOfRuns)
```

## Architecture overview
<img src="https://ll-srv-web.livinglab.local:3000/Johann.Haselberger/MINKA/raw/branch/master/assets/minka_overview2.png">

## Usage
### Installation
```
!pip install --upgrade PyYAML==4.2b4 optuna sacred python-telegram-bot dnspython loguru wandb pymongo
```

### Preparation
* In order to store the data, a mongoDB is required. You can host your own free one here: https://cloud.mongodb.com
* If you want to use the w&b interface, an account is needed. Create one here: https://app.wandb.ai

### Define your optimization problem
Minka uses a very simple combination of a configuration json file and the actual optimization task, represented as a single class.

#### optimization class
optimization template:
```python
class yourOptimizationTask:
    def __init__(self):
        pass
    
    def prepare(self):
        pass
    
    def run(self, config):
        x = config['x']
        result = (x - 2) ** 2
        
        evalMetrics = {
            'error': result
        }
        
        logArrays = {
            'someValues': [1,2,3,4]
        }
        
        return result, evalMetrics, logArrays
    
    def cleanup(self):
        pass

```

#### parameter configuration
content of the configuration.json file:
```json

{
    "comment": "parameter types: fix, categorical, discrete_uniform, int, loguniform, uniform",
    "parameters": {
        "batchSize": {"type": "fix","values": 256},
        "epochs": {"type": "fix","values": 75},
        "x": {"type": "categorical","values": [11,22,26]}
    }
}

```

## Screenshots
<img src="https://ll-srv-web.livinglab.local:3000/Johann.Haselberger/MINKA/raw/branch/master/assets/omni.png">
<img src="https://ll-srv-web.livinglab.local:3000/Johann.Haselberger/MINKA/raw/branch/master/assets/wandb.png">
