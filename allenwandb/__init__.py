from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from allennlp.training import TrainerCallback, GradientDescentTrainer
from allennlp.data import TensorDict

import os, json
import wandb
import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _internet_available(timeout=10):
    import subprocess
    try:
        res = subprocess.run(['wget', '-q', '--spider', 'google.com'], timeout=timeout)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False

@TrainerCallback.register('wandb')
class WandbLoggerCallback(TrainerCallback):
    def __init__(
        self, 
        project: str = None,
        entity: str = None,
        name: str = None,
        group: str = None,
        tags: List[str] = [],
        sync_tensorboard: bool = False,
        dir: str = None,
        mode='run',
    ):
        self._init_args = dict(
            name=name,
            project=project,
            entity=entity,
            group=group, 
            tags=tags,
            dir=dir,
            sync_tensorboard=sync_tensorboard,
            mode=mode if _internet_available() else 'dryrun',
        )

    @property
    def serialization_dir(self):
        return Path(self.trainer._serialization_dir)

    def on_start(
        self, trainer: GradientDescentTrainer, is_primary: bool = True, **kwargs
    ) -> None:
        self.trainer = trainer
        if not is_primary:
            return
        
        with (self.serialization_dir / "config.json").open() as f:
            config = json.load(f)
        
        self.run = wandb.init(**self._init_args, config=config)
        self.run.watch(trainer.model, log="all")

    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not is_primary:
            return

        self.run.log(batch_metrics)

    def on_epoch(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if not is_primary:
            return

        self.run.log(metrics)

    def on_end(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if not is_primary:
            return

        print(metrics)
        self.run.log(metrics)
        self.run.summary.update(metrics)

        for f in self.serialization_dir.glob('**/.lock'):
            f.unlink(missing_ok=True)

        artifact = wandb.Artifact('model', type='model')
        artifact.add_dir(str(self.serialization_dir))

        self.run.log_artifact(artifact)
        # self.run.finish()