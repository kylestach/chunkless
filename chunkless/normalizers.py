from collections import OrderedDict
import fnmatch
from typing import Any, Callable, Iterable, List, Literal, Mapping, Sequence, Tuple
import jax
from jax.tree_util import (
    tree_flatten_with_path, tree_unflatten,
    GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
)
import torch



def keystr(keypath, separator="/"):
    def _keystr(k):
        if isinstance(k, jax.tree_util.SequenceKey):
            return str(k.idx)
        elif isinstance(k, jax.tree_util.DictKey):
            return str(k.key)
        elif isinstance(k, jax.tree_util.GetAttrKey):
            return str(k.name)
        elif isinstance(k, jax.tree_util.FlattenedIndexKey):
            return str(k.key)
        else:
            raise ValueError(f"Unknown key type: {type(k)}")

    return separator.join([_keystr(k) for k in keypath])


def assign_labels(data, label_patterns: List[Tuple[str, str]]):
    used_rules = set()

    def _assign_label(keypath, _):
        for pattern, label in label_patterns:
            if fnmatch.fnmatch(keystr(keypath), pattern):
                used_rules.add(pattern)
                return label
        raise ValueError(f"No label found for {keystr(keypath)}")

    return jax.tree_util.tree_map_with_path(_assign_label, data)

def get_element_at_path(tree, path):
    node = tree
    for key_entry in path:
        if isinstance(key_entry, GetAttrKey):
            key = key_entry.key
            if hasattr(node, key):
                node = getattr(node, key)
            else:
                return None
        elif isinstance(key_entry, DictKey):
            key = key_entry.key
            try:
                node = node[key]
            except (KeyError, TypeError):
                return None
        elif isinstance(key_entry, SequenceKey):
            index = key_entry.idx
            try:
                node = node[index]
            except (IndexError, TypeError):
                return None
        elif isinstance(key_entry, FlattenedIndexKey):
            index = key_entry.idx
            if hasattr(node, '__getitem__'):
                try:
                    node = node[index]
                except (IndexError, TypeError):
                    return None
            else:
                return None
        else:
            return None
    return node

def collect_elements(tree, *rest):
    paths, _ = tree_flatten_with_path(tree)
    new_leaves = []
    for path, value in paths:
        collected_elements = [value]
        for r in rest:
            elem = get_element_at_path(r, path)
            collected_elements.append(elem)
        collected_elements = tuple(collected_elements)
        new_leaves.append(collected_elements)
    treedef = jax.tree_util.tree_structure(tree)
    new_tree = tree_unflatten(treedef, new_leaves)
    return new_tree

def tree_map_with_none(fn, tree, *rest):
    paths, _ = tree_flatten_with_path(tree)
    new_leaves = []
    for path, value in paths:
        collected_elements = [value]
        for r in rest:
            elem = get_element_at_path(r, path)
            collected_elements.append(elem)
        collected_elements = tuple(collected_elements)
        new_leaves.append(collected_elements)
    treedef = jax.tree_util.tree_structure(tree)

    new_leaves = list(map(lambda x: fn(*x), new_leaves))
    new_tree = tree_unflatten(treedef, new_leaves)
    return new_tree


class Normalizer:
    def __init__(
        self,
        stats,
        normalize_rules: Sequence[Tuple[str, Literal["mean_std", "min_max", "none"]]]
    ):
        self.stats = jax.tree_map(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x, stats)
        self.normalize_fns = {
            "mean_std": lambda x, stats: (x - stats["mean"]) / stats["std"],
            "min_max": lambda x, stats: (x - stats["min"]) / (stats["max"] - stats["min"]) * 2 - 1,
            "none": lambda x, _: x,
        }
        self.unnormalize_fns = {
            "mean_std": lambda x, stats: x * stats["std"] + stats["mean"],
            "min_max": lambda x, stats: (x + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"],
            "none": lambda x, _: x,
        }
        self.normalize_rules = normalize_rules

    def _apply_fns(self, data, fns, key=None):
        if key is not None:
            data = {key: data}

        labels = assign_labels(data, self.normalize_rules)
        data = tree_map_with_none(lambda x, s, l: fns[l](x, s), data, self.stats, labels)

        if key is not None:
            data = data[key]
        
        return data
    
    def normalize(self, data, key=None):
        return self._apply_fns(data, self.normalize_fns, key)

    def unnormalize(self, data, key=None):
        return self._apply_fns(data, self.unnormalize_fns, key)