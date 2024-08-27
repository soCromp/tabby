import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import os
from be_great.multihead_models import MOEModelForCausalLM

def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class GReaTTrainer(Trainer):
    """GReaT Trainer

    Overwrites the get_train_dataloader methode of the HuggingFace Trainer to not remove the "unused" columns -
    they are needed later!
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = (
            self.train_dataset
        )  # self._remove_unused_columns(self.train_dataset, description="training")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )

    def get_eval_dataloader(self, dataset=None) -> DataLoader:
        if dataset == None:
            if self.eval_dataset is None:
                raise ValueError("Trainer: eval requires an eval_dataset.")
            else:
                dataset = self.eval_dataset

        data_collator = self.data_collator
        eval_dataset = (
            dataset
        )  # self._remove_unused_columns(self.train_dataset, description="training")
        eval_sampler = self._get_eval_sampler(dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=eval_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )
        
        
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        print(output_dir)
        
        if os.path.exists(output_dir):
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
