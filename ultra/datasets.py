import os

import torch
from torch_geometric.data import Data

from ultra.tasks import build_relation_graph


def _load_triplets(path, entity2id, relation2id):
    triplets = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 'head relation tail'"
                )

            head, relation, tail = fields
            if head not in entity2id:
                entity2id[head] = len(entity2id)
            if tail not in entity2id:
                entity2id[tail] = len(entity2id)
            if relation not in relation2id:
                relation2id[relation] = len(relation2id)

            triplets.append(
                (entity2id[head], entity2id[tail], relation2id[relation])
            )
    return triplets


def _target_tensors(triplets):
    if not triplets:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(
            (0,), dtype=torch.long
        )
    target_edge_index = torch.tensor(
        [(head, tail) for head, tail, _ in triplets], dtype=torch.long
    ).t()
    target_edge_type = torch.tensor(
        [relation for _, _, relation in triplets], dtype=torch.long
    )
    return target_edge_index, target_edge_type


class LocalTransductiveDataset:
    """Load one local KG from ``root/name/{train,valid,test}.txt``."""

    def __init__(self, root, name):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.name = name
        self.entity2id = {}
        self.relation2id = {}

        graph_dir = os.path.join(self.root, name)
        split_paths = {
            split: os.path.join(graph_dir, f"{split}.txt")
            for split in ("train", "valid", "test")
        }
        missing = [path for path in split_paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "Missing dataset split(s): " + ", ".join(missing)
            )

        split_triplets = {
            split: _load_triplets(path, self.entity2id, self.relation2id)
            for split, path in split_paths.items()
        }
        num_nodes = len(self.entity2id)
        num_forward_relations = len(self.relation2id)

        train_index, train_type = _target_tensors(split_triplets["train"])
        valid_index, valid_type = _target_tensors(split_triplets["valid"])
        test_index, test_type = _target_tensors(split_triplets["test"])

        fact_index = torch.cat([train_index, train_index.flip(0)], dim=1)
        fact_type = torch.cat(
            [train_type, train_type + num_forward_relations], dim=0
        )
        num_relations = num_forward_relations * 2

        def make_graph(target_index, target_type):
            graph = Data(
                edge_index=fact_index,
                edge_type=fact_type,
                target_edge_index=target_index,
                target_edge_type=target_type,
                num_nodes=num_nodes,
                num_relations=num_relations,
            )
            return build_relation_graph(graph)

        self.train_data = [make_graph(train_index, train_type)]
        self.valid_data = [make_graph(valid_index, valid_type)]
        self.test_data = [make_graph(test_index, test_type)]

    def __repr__(self):
        return f"LocalTransductiveDataset(name={self.name!r})"


class JointDataset:
    """Minimal multi-graph wrapper used by the self-evolution trainer."""

    def __init__(self, root, graphs):
        if not graphs:
            raise ValueError("dataset.graphs must contain at least one graph name")
        self.root = os.path.abspath(os.path.expanduser(root))
        self.graphs = [
            LocalTransductiveDataset(self.root, graph_name)
            for graph_name in graphs
        ]
        self._data = (
            [graph.train_data[0] for graph in self.graphs],
            [graph.valid_data[0] for graph in self.graphs],
            [graph.test_data[0] for graph in self.graphs],
        )
        self.data = self._data

