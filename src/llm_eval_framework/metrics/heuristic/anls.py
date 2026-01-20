import math
import warnings
from typing import Any, Tuple, List, Union

from llm_eval_framework.metrics.base import BaseMetric, MetricResult

try:
    from munkres import Munkres, make_cost_matrix
except ImportError:
    Munkres = None
    make_cost_matrix = None


class ANLSMetric(BaseMetric):
    """ANLS* metric for document processing and structured data comparison.

    Supports nested structures: tuples (1-of-n ground truths), lists (Hungarian matching),
    dicts (key-wise matching), and primitives (Levenshtein similarity).
    """

    THRESHOLD = 0.5

    def __init__(self):
        """Initialize ANLSMetric."""
        super().__init__(name="anls")

        if Munkres is None:
            raise ImportError(
                "munkres required for ANLS. Install with: pip install munkres"
            )

    def score(self, output: Any, reference: Any) -> MetricResult:
        """Calculate ANLS score between output and reference.

        Args:
            output: The model output (can be nested structures)
            reference: The reference/ground truth (can be nested structures)

        Returns:
            MetricResult with ANLS score (0.0 to 1.0)
        """
        # Handle classical QA dataset compatibility
        if (
            isinstance(reference, list)
            and all(isinstance(x, str) for x in reference)
            and isinstance(output, str)
        ):
            warnings.warn("Converting list reference to tuple for QA compatibility")
            reference = tuple(reference)

        gt_tree = self._make_tree(reference, is_gt=True)
        pred_tree = self._make_tree(output, is_gt=False)

        nls_list, _ = gt_tree.nls_list(pred_tree)
        length = gt_tree.pairwise_len(pred_tree)
        result = (sum(nls_list) / length) if length > 0 else 1.0

        return MetricResult(value=result)

    def _make_tree(self, obj: Any, is_gt: bool) -> "ANLSNode":
        """Create an ANLS tree node from an object."""
        if isinstance(obj, tuple):
            return TupleNode(obj, is_gt)
        elif isinstance(obj, list):
            return ListNode(obj, is_gt)
        elif isinstance(obj, dict):
            return DictNode(obj, is_gt)
        elif obj is None or obj == "" or obj == [] or obj == {}:
            return NoneNode()
        elif isinstance(obj, (str, float, int, bool)):
            return LeafNode(obj)
        else:
            raise ValueError(f"Unsupported type {type(obj)} for ANLS")


class ANLSNode:
    """Base class for ANLS tree nodes."""

    def __init__(self, obj: Any):
        self.obj = obj

    def __len__(self) -> int:
        return 1

    def pairwise_len(self, other: "ANLSNode") -> int:
        return max(len(self), len(other))

    def nls_list(self, other: "ANLSNode") -> Tuple[List[float], Any]:
        return [0.0], self.obj


class TupleNode(ANLSNode):
    """ANLS node for tuples (1-of-n ground truths)."""

    def __init__(self, obj: tuple, is_gt: bool):
        if not is_gt:
            raise ValueError("Tuples only allowed in ground truth")
        super().__init__(obj)
        # Create a temporary ANLSMetric instance to use _make_tree
        metric = ANLSMetric()
        self.children = [metric._make_tree(x, is_gt) for x in obj]

    def __len__(self):
        return max(len(x) for x in self.children)

    def nls_list(self, other: ANLSNode) -> Tuple[List[float], Any]:
        best_nls = []
        best_gt = self.obj
        best_score = -1

        for child in self.children:
            nls_list, chosen_gt = child.nls_list(other)
            length = child.pairwise_len(other)
            avg_score = sum(nls_list) / length if length > 0 else 1.0

            # Prefer exact matches
            if other.obj == chosen_gt:
                avg_score = math.nextafter(avg_score, float("inf"))

            if avg_score > best_score:
                best_score = avg_score
                best_nls = nls_list
                best_gt = chosen_gt

        return best_nls, best_gt


class ListNode(ANLSNode):
    """ANLS node for lists (Hungarian matching)."""

    def __init__(self, obj: list, is_gt: bool):
        super().__init__(obj)
        # Create a temporary ANLSMetric instance to use _make_tree
        metric = ANLSMetric()
        self.children = [metric._make_tree(x, is_gt) for x in obj]

    def __len__(self):
        return sum(len(x) for x in self.children)

    def pairwise_len(self, other: ANLSNode) -> int:
        if not isinstance(other, ListNode):
            return max(len(self), len(other))

        # Hungarian matching to get optimal pairing length
        if not self.children or not other.children:
            return len(self) + len(other)

        # Compute cost matrix
        costs = []
        for gt_child in self.children:
            row = []
            for pred_child in other.children:
                nls_list, _ = gt_child.nls_list(pred_child)
                length = gt_child.pairwise_len(pred_child)
                avg_score = sum(nls_list) / length if length > 0 else 1.0
                row.append(1.0 - avg_score)  # Convert to cost
            costs.append(row)

        # Apply Hungarian algorithm
        cost_matrix = make_cost_matrix(costs)
        indices = Munkres().compute(cost_matrix)

        # Calculate total length
        total_len = 0
        matched_gt = set()
        matched_pred = set()

        for row, col in indices:
            total_len += self.children[row].pairwise_len(other.children[col])
            matched_gt.add(row)
            matched_pred.add(col)

        # Add unmatched elements
        for i, child in enumerate(self.children):
            if i not in matched_gt:
                total_len += len(child)

        for i, child in enumerate(other.children):
            if i not in matched_pred:
                total_len += len(child)

        return total_len

    def nls_list(self, other: ANLSNode) -> Tuple[List[float], Any]:
        if not isinstance(other, ListNode):
            return [0.0], self.obj

        if not self.children or not other.children:
            return [0.0], self.obj

        # Hungarian matching
        costs = []
        all_nls = []
        all_gts = []

        for gt_child in self.children:
            row_costs = []
            row_nls = []
            row_gts = []

            for pred_child in other.children:
                nls_list, chosen_gt = gt_child.nls_list(pred_child)
                length = gt_child.pairwise_len(pred_child)
                avg_score = sum(nls_list) / length if length > 0 else 1.0

                if pred_child.obj == chosen_gt:
                    avg_score = math.nextafter(avg_score, float("inf"))

                row_costs.append(1.0 - avg_score)
                row_nls.append(nls_list)
                row_gts.append(chosen_gt)

            costs.append(row_costs)
            all_nls.append(row_nls)
            all_gts.append(row_gts)

        # Apply Hungarian algorithm
        cost_matrix = make_cost_matrix(costs)
        indices = Munkres().compute(cost_matrix)

        # Collect results
        result_nls = []
        result_gt = []
        matched_pred_idx = set()

        for row, col in indices:
            result_nls.extend(all_nls[row][col])
            result_gt.append(all_gts[row][col])
            matched_pred_idx.add(col)

        # Add unmatched ground truth items
        unmatched_gt = [
            i for i in range(len(self.children)) if i not in {r for r, c in indices}
        ]
        for i in unmatched_gt:
            result_gt.append(self.children[i].obj)

        # Sort result_gt by original prediction order
        sorted_pairs = [
            (gt, col) for (_, col), gt in zip(indices, result_gt[: len(indices)])
        ]
        sorted_pairs.sort(key=lambda x: x[1])
        sorted_gt = [gt for gt, _ in sorted_pairs] + result_gt[len(indices) :]

        return result_nls, sorted_gt


class DictNode(ANLSNode):
    """ANLS node for dictionaries."""

    def __init__(self, obj: dict, is_gt: bool):
        super().__init__(obj)
        # Create a temporary ANLSMetric instance to use _make_tree
        metric = ANLSMetric()
        self.children = {k: metric._make_tree(v, is_gt) for k, v in obj.items()}

    def __len__(self):
        return sum(len(x) for x in self.children.values())

    def pairwise_len(self, other: ANLSNode) -> int:
        if not isinstance(other, DictNode):
            return max(len(self), len(other))

        total_len = 0
        for k in self.children.keys() | other.children.keys():
            self_child = self.children.get(k, NoneNode())
            other_child = other.children.get(k, NoneNode())
            total_len += self_child.pairwise_len(other_child)

        return total_len

    def nls_list(self, other: ANLSNode) -> Tuple[List[float], Any]:
        if not isinstance(other, DictNode):
            return [0.0], self.obj

        result_nls = []
        result_gt = {}

        for k in list(self.children.keys()) + [
            k for k in other.children.keys() if k not in self.children
        ]:
            self_child = self.children.get(k, NoneNode())
            other_child = other.children.get(k, NoneNode())

            # Skip hallucinated None keys
            if (
                k not in self.children
                and k in other.children
                and _is_none_like(other_child.obj)
            ):
                continue

            nls_list, chosen_gt = self_child.nls_list(other_child)
            result_nls.extend(nls_list)

            # Don't include None keys that weren't in prediction
            if not (_is_none_like(chosen_gt) and k not in other.children):
                result_gt[k] = chosen_gt

        return result_nls, result_gt


class NoneNode(ANLSNode):
    """ANLS node for None/empty values."""

    def __init__(self):
        super().__init__(None)

    def nls_list(self, other: ANLSNode) -> Tuple[List[float], Any]:
        if _is_none_like(other.obj):
            return [1.0], other.obj
        return [0.0], self.obj


class LeafNode(ANLSNode):
    """ANLS node for primitive values."""

    def __init__(self, obj: Union[str, int, float, bool]):
        super().__init__(obj)

    def nls_list(self, other: ANLSNode) -> Tuple[List[float], Any]:
        if not isinstance(other, LeafNode):
            return [0.0], self.obj

        str1 = " ".join(str(self.obj).strip().lower().split())
        str2 = " ".join(str(other.obj).strip().lower().split())

        if not str1 and not str2:
            return [1.0], self.obj
        if not str1 or not str2:
            return [0.0], self.obj

        # Levenshtein distance
        dist = self._levenshtein(str1, str2)
        max_len = max(len(str1), len(str2))
        similarity = 1.0 - (dist / max_len)

        # Apply threshold
        score = similarity if similarity >= ANLSMetric.THRESHOLD else 0.0
        return [score], self.obj

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = list(range(len(s1) + 1))
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(
                        1 + min(distances[i1], distances[i1 + 1], new_distances[-1])
                    )
            distances = new_distances

        return distances[-1]


def _is_none_like(obj: Any) -> bool:
    """Check if object is None-like (None, empty string, empty list, empty dict)."""
    return obj in (None, "", [], {})
