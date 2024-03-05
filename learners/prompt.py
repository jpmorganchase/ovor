# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

from .default import NormalNN
import models

class Prompt(NormalNN):
    def __init__(self, out_dim, args, logger):
        self.prompt_param = [args.tasks, args.prompt_param]
        super().__init__(out_dim, args, logger)

    def _get_learnable_params(self):
        return list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())

    def _create_model(self):
        pass

class CODAPrompt(Prompt):
    def _create_model(self):
        return models.__dict__[self.model_type].__dict__[self.model_name](
            out_dim=self.out_dim, prompt_flag='coda', prompt_param=self.prompt_param
        )

class DualPrompt(Prompt):
    def _create_model(self):
        return models.__dict__[self.model_type].__dict__[self.model_name](
            out_dim=self.out_dim, prompt_flag='dual', prompt_param=self.prompt_param
        )

class L2P(Prompt):
    def _create_model(self):
        return models.__dict__[self.model_type].__dict__[self.model_name](
            out_dim=self.out_dim, prompt_flag='l2p', prompt_param=self.prompt_param
        )

class OnePrompt(Prompt):
    def _create_model(self):
        return models.__dict__[self.model_type].__dict__[self.model_name](
            out_dim=self.out_dim, prompt_flag='oneprompt', prompt_param=self.prompt_param
        )