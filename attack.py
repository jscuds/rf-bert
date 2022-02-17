from typing import List

import textattack
import torch

from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attacker import Attacker

from allennlp.modules.elmo import batch_to_ids
from models import ElmoClassifier
from mosestokenizer import MosesTokenizer

from dataloaders.helpers import elmo_tokenize_and_pad

# TODO: make this work with an arbitrary model?
# TODO: support datasets besides rotten tomatoes
# TODO: make this work for things besides binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: argparse for all this stuff.
num_examples = 1000
attack_in_parallel = False
# I have one model saved that's just fine-tuned from scratch (with no M matrix):
# model_path, model_is_retrofitted = 'models/finetune_elmo_single_sentence_2022-02-15-1437/final.pth', False
# And I have another model that's fine-tuned after retrofitting:
# model_path, model_is_retrofitted = 'models/finetune_elmo_single_sentence_2022-02-15-1457/final.pth', True

model_path, model_is_retrofitted = 'models/finetune_elmo_2022-02-17-1226/final.pth', True

class ElmoModelWrapper(textattack.models.wrappers.ModelWrapper):
    tokenizer: MosesTokenizer
    model: torch.nn.Module
    max_length: int
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.tokenizer = elmo_tokenizer = MosesTokenizer('en', no_escape=True)
        self.max_length = 40

    def __call__(self, text_input_list: List[str]):
        outputs = []
        text_ids = torch.stack([
            elmo_tokenize_and_pad(self.tokenizer, t, self.max_length)
            for t in text_input_list
        ]).to(device)
        # Seq length or batch size?
        with torch.no_grad():
            probs = self.model(text_ids)
        # TextAttack expects probabilities for each class,
        # so we 
        probs = torch.stack([1-probs, probs], dim=-1)
        return probs.cpu().numpy()

def main():
    # Create model and load from weights
    model = ElmoClassifier(
        num_output_representations=1,
        sentence_pair=False,
        m_transform=model_is_retrofitted, # only true if attacking retrofitted model
    )
    model_weights = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(model_weights['model'], strict=False)
    print('*** loaded model weights into model, missing_keys =', missing_keys)
    print('*** loaded model weights into model, unexpected_keys =', unexpected_keys)
    model = model.to(device)
    model_wrapper = ElmoModelWrapper(model)

    # Run attack
    dataset = HuggingFaceDataset("rotten_tomatoes", split="test")
    attack = TextFoolerJin2019.build(model_wrapper)
    attack_args = textattack.AttackArgs(
        num_examples=num_examples, parallel=attack_in_parallel
    )
    attacker = Attacker(attack, dataset, attack_args=attack_args)
    attacker.attack_dataset()

if __name__ == '__main__': main()
