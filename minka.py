from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import TelegramObserver
import optuna
import numpy as np
import wandb
from wandb.keras import WandbCallback
import json
import pymongo
import copy
from loguru import logger


class tempParam:
    def __init__(self, value):
        self.value = value


class minka:
    def __init__(
        self,
        targetClass,
        configPath,
        expName="minka_ex",
        db="minka_db",
        mongoURL="XXX",
        attachMongoDB=False,
        attachTelegram=False,
        attachWandB=False,
    ):
        self.targetClass = targetClass
        self.expName = expName
        self.db = db
        self.mongoURL = mongoURL
        self.attachMongoDB = attachMongoDB
        self.attachTelegram = attachTelegram
        self.attachWandB = attachWandB
        self.configPath = configPath

        with open(self.configPath) as handle:
            jConf = json.loads(handle.read())
        self.jParams = jConf["parameters"]

    def _logArrayToEx(self, ex, array, name):
        assert len(array.shape) == 1, "right now only 1D arrays are supported"
        for i in range(array.size):
            ex.log_scalar(name, array[i], i)

    def _isExPresent(self, config):
        _client = pymongo.MongoClient(self.mongoURL)
        _db = _client[self.db]
        _runsColl = _db["runs"]

        query = {"experiment.name": self.expName, "status": "COMPLETED"}
        for c in config:
            _v = config[c]
            _v = int(_v) if type(_v) == np.int64 else _v
            query["config." + c] = _v
        # del query['config.seed']

        _l = list(_runsColl.find(query, {"result": 1}))

        return (True, _l[0]["result"]) if len(_l) > 0 else (False, 0)

    def _objectiveWrapper(self, trial):

        # define all needed parameters
        _conf = {}
        _pp = {}
        jParams = self.jParams
        for param in jParams:
            if jParams[param]["type"] == "fix":
                _conf[param] = jParams[param]["values"]
            elif jParams[param]["type"] == "categorical":
                tt = tempParam(jParams[param]["values"])
                _conf[param] = trial.suggest_categorical(param, tt.value)
            elif jParams[param]["type"] == "discrete_uniform":
                _conf[param] = trial.suggest_discrete_uniform(
                    param, jParams[param]["values"][0], jParams[param]["values"][1]
                )
            elif jParams[param]["type"] == "int":
                _conf[param] = trial.suggest_int(
                    param, jParams[param]["values"][0], jParams[param]["values"][1]
                )
            elif jParams[param]["type"] == "loguniform":
                _conf[param] = trial.suggest_loguniform(
                    param, jParams[param]["values"][0], jParams[param]["values"][1]
                )
            elif jParams[param]["type"] == "uniform":
                _conf[param] = trial.suggest_uniform(
                    param, jParams[param]["values"][0], jParams[param]["values"][1]
                )

        if self.attachMongoDB:
            pre, preResult = self._isExPresent(_conf)
            if pre:
                logger.debug("-----> Skipping this ex")
                return preResult

        # create a new experiment and attach mongoDB observer
        ex = Experiment(self.expName, interactive=True)

        if self.attachMongoDB:
            ex.observers.append(
                MongoObserver.create(url=self.mongoURL, db_name=self.db)
            )

        if self.attachTelegram:
            telegram_obs = TelegramObserver.from_config("./telegramConfig.json")
            ex.observers.append(telegram_obs)

        if self.attachWandB:
            wandb.init(project=self.expName, reinit=True)

        ex.add_config(_conf)
        if self.attachWandB:
            _c = copy.deepcopy(_conf)
            for key in _c:
                _c[key] = int(_c[key]) if type(_c[key]) == np.int64 else _c[key]
            wandb.config.update(_c)
            # wandb.config.update({'affe': 1})

        # the main experiment function
        @ex.main
        def objective():

            callbacks = [WandbCallback()] if self.attachWandB else []
            result, evalMetrics, logArrays = self.targetClass.run(_conf, callbacks)

            for key in logArrays.keys():
                self._logArrayToEx(ex, logArrays[key], key)

            # store all needed evaluation metrics
            ex.info["evalMetrics"] = evalMetrics

            if self.attachWandB:
                wandb.log(evalMetrics)
                # pass

            return result

        # run the single experiment
        r = ex.run()

        # for the optimization part, we have to return a single float
        return r.result

    def opt(self, numberOfTrials=1):
        self.targetClass.prepare()
        study = optuna.create_study()
        study.optimize(self._objectiveWrapper, n_trials=numberOfTrials)
        self.targetClass.cleanup()
        print("Best params: " + str(study.best_params))
        return study
