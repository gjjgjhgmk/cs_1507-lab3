import torch
import numpy as np
from parse import args
import torch.nn as nn
import math

class OurAttackClient(nn.Module):
    def __init__(self, target_items):
        super().__init__()
        ...

    def eval_(self, _items_emb):
        return None, None
    
    def compute_k_popularities(self, k, items_emb):
        norms = torch.norm(items_emb, dim=1)
        self.k_popularities = torch.argsort(norms, descending=True)[:k]

    def train_(self, items_emb):
        with torch.no_grad():
            if self.global_rounds == args.attack_round:
                self.compute_k_popularities(args.k, items_emb)
                top_k_embedding = items_emb[self.k_popularities]
                average_top_k_embedding = torch.mean(top_k_embedding, axis=0)
                self.target_model = items_emb.clone()
                self.target_model[self.target_items] = average_top_k_embedding * 10
            if self.global_rounds < args.attack_round:
                self.global_rounds += 1
                return None, None, None
            self.global_rounds += 1
            items_emb_model_update = (self.target_model - items_emb) * args.alpha
            chosen_items = torch.argsort(torch.norm(items_emb_model_update, dim=1), descending=True)[:args.items_limit - len(self.target_items)]
            # erase target items from chosen_items
            chosen_items = torch.tensor(list(set(chosen_items.tolist()) - set(self.target_items))).to(args.device)
            # add target items to chosen_items
            chosen_items = torch.cat((chosen_items, torch.tensor(self.target_items).to(args.device)), dim=0)
        return chosen_items, items_emb_model_update[chosen_items], None
