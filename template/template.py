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