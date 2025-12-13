"""
label_propagation.py

Implementation of the Label Propagation algorithm for community detection
on an undirected graph G = (V, E).

This version enforces that labels are always non-negative integers, even if
the input vertices are strings or other hashable types. It also provides a
convenience method to return an n×1 vector of community labels in the
original vertex order.
"""

from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple


class community_detection:
    """Label Propagation community detection on an undirected graph.

    Each vertex starts with a unique integer label. At every iteration,
    vertices are processed in random order and adopt the most frequent
    label among their neighbors (ties are broken at random). The process
    stops when either:

      * The maximum number of iterations is reached, or
      * The fraction of vertices that change label in one iteration is
        less than or equal to the threshold ``epsilon``.

    Attributes:
        max_iters: Maximum number of iterations to perform.
        epsilon: Threshold for the fraction of label changes per
            iteration below which the algorithm will stop early.
    """

    def __init__(
        self,
        vertices: Iterable[Hashable],
        edges: Iterable[Tuple[Hashable, Hashable]],
        max_iters: int,
        epsilon: float,
    ) -> None:
        """Initialize the community_detection algorithm with a graph G = (V, E).

        Args:
            vertices: Iterable of hashable objects representing the
                vertices V of the graph.
            edges: Iterable of 2-tuples (u, v) representing undirected
                edges E of the graph, where u and v are vertices in V.
            max_iters: Maximum number of label propagation iterations
                to perform. Must be a positive integer.
            epsilon: Threshold for the fraction of vertices that change
                label in a single iteration below which the algorithm
                will stop early. Must be in the range [0.0, 1.0].

        Raises:
            TypeError: If the input types are incorrect.
            ValueError: If parameter values are invalid or edges refer
                to vertices not in V, or if vertices are not unique.
        """
        self._check_init_types(vertices, edges, max_iters, epsilon)

        self.max_iters: int = max_iters
        self.epsilon: float = float(epsilon)

        # Keep the original order of vertices for vector export.
        self._vertex_list: List[Hashable] = list(vertices)
        self._vertices: Set[Hashable] = set(self._vertex_list)

        # Ensure vertices are unique.
        if len(self._vertex_list) != len(self._vertices):
            raise ValueError("Vertices must be unique.")

        # Map each vertex to a unique non-negative integer index.
        # These indices will also serve as the initial labels.
        self._vertex_to_index: Dict[Hashable, int] = {
            v: i for i, v in enumerate(self._vertex_list)
        }

        self._adjacency: Dict[Hashable, Set[Hashable]] = self._build_adjacency(
            self._vertices, edges
        )

        # Mapping from vertex -> integer label (non-negative).
        self._labels: Dict[Hashable, int] = {}

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _check_init_types(
        self,
        vertices: Iterable[Hashable],
        edges: Iterable[Tuple[Hashable, Hashable]],
        max_iters: int,
        epsilon: float,
    ) -> None:
        """Validate types and ranges of the initializer arguments.

        This helper checks that:
          * ``max_iters`` is an integer and strictly positive.
          * ``epsilon`` is a float (or int) within [0.0, 1.0].
          * ``vertices`` and ``edges`` are iterable.

        Args:
            vertices: Candidate iterable of vertices.
            edges: Candidate iterable of edges.
            max_iters: Maximum number of iterations requested.
            epsilon: Early-stopping threshold on fraction of label changes.

        Raises:
            TypeError: If ``max_iters`` or ``epsilon`` have invalid types,
                or if ``vertices`` / ``edges`` are not iterable.
            ValueError: If ``max_iters`` is non-positive or ``epsilon``
                is out of the range [0.0, 1.0].
        """
        # Check max_iters
        if not isinstance(max_iters, int):
            raise TypeError("max_iters must be an integer.")
        if max_iters <= 0:
            raise ValueError("max_iters must be strictly positive.")

        # Check epsilon
        if not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a float or int.")
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("epsilon must be in the range [0.0, 1.0].")

        # Check iterability of vertices and edges
        try:
            iter(vertices)
        except TypeError as exc:
            raise TypeError("vertices must be an iterable of hashable objects.") from exc

        try:
            iter(edges)
        except TypeError as exc:
            raise TypeError("edges must be an iterable of 2-tuples (u, v).") from exc

    def _build_adjacency(
        self,
        vertices: Set[Hashable],
        edges: Iterable[Tuple[Hashable, Hashable]],
    ) -> Dict[Hashable, Set[Hashable]]:
        """Build an adjacency list representation of the undirected graph.

        Args:
            vertices: Set of vertices in the graph.
            edges: Iterable of 2-tuples (u, v) representing undirected edges.

        Returns:
            Dictionary mapping each vertex to a set of its neighbors.

        Raises:
            TypeError: If any edge is not a 2-tuple.
            ValueError: If an edge references a vertex not in ``vertices``.
        """
        adjacency: Dict[Hashable, Set[Hashable]] = {v: set() for v in vertices}

        for edge in edges:
            if not (isinstance(edge, tuple) and len(edge) == 2):
                raise TypeError("Each edge must be a 2-tuple (u, v).")

            u, v = edge

            if u not in vertices or v not in vertices:
                raise ValueError(
                    f"Edge ({u}, {v}) references vertex not present in vertices."
                )

            if u == v:
                # Ignore self-loops.
                continue

            adjacency[u].add(v)
            adjacency[v].add(u)

        return adjacency

    def _initialize_labels(self) -> None:
        """Initialize vertex labels.

        Each vertex starts with a unique non-negative integer label.
        The initial label assigned to vertex ``v`` is its index in the
        original vertex list (0..n-1).

        Returns:
            self._labels: dict
            Maps each verticies to a unique initial label
        """
        self._labels = {v: self._vertex_to_index[v] for v in self._vertex_list}
        return  self._labels

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def run(self, seed: Optional[int] = None) -> Dict[Hashable, int]:
        """Run the Label Propagation algorithm.

        This method initializes labels, performs up to ``max_iters``
        iterations of label propagation, and optionally stops early
        when the fraction of label changes drops below ``epsilon``.

        Args:
            seed: Optional seed for the random number generator to make
                the algorithm deterministic across runs.

        Returns:
            Dictionary mapping each vertex to its final integer label
            (non-negative). Vertices with the same label are in the
            same community.
        """
        if seed is not None:
            random.seed(seed)

        self._initialize_labels()

        if not self._vertices:
            # Empty graph
            return {}

        for _ in range(self.max_iters):
            changed_fraction = self._single_iteration()
            if changed_fraction <= self.epsilon:
                break

        # Return a copy of the labels as vertex -> int.
        return dict(self._labels)

    def _single_iteration(self) -> float:
        """Perform a single iteration of label propagation.

        In one iteration, vertices are visited in random order. Each
        non-isolated vertex adopts the most frequent label among its
        neighbors (with random tie-breaking). The method counts how
        many vertices changed labels in this pass.

        Returns:
            Fraction of vertices whose label changed in this iteration.
            This is a float in the range [0.0, 1.0].
        """
        vertices_list: List[Hashable] = list(self._vertices)
        random.shuffle(vertices_list)

        num_changed = 0
        total_vertices = len(vertices_list)

        for v in vertices_list:
            neighbors = self._adjacency[v]

            if not neighbors:
                # Isolated vertex; label cannot change.
                continue

            # Count labels among neighbors
            label_counts: Dict[int, int] = {}
            for u in neighbors:
                label = self._labels[u]
                label_counts[label] = label_counts.get(label, 0) + 1

            # Most frequent label(s)
            max_count = max(label_counts.values())
            best_labels = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

            # Break ties randomly
            new_label = random.choice(best_labels)

            if new_label != self._labels[v]:
                self._labels[v] = new_label
                num_changed += 1

        if total_vertices == 0:
            return 0.0

        return num_changed / total_vertices

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_communities(self) -> Dict[int, Set[Hashable]]:
        """Group vertices by their final labels into communities.

        This method constructs communities from the current label
        assignments. It assumes labels have already been computed
        (e.g., by calling :meth:`run`). If called beforehand, it
        reflects whatever labels are currently stored (for example,
        just the initial labels if :meth:`run` has not been called).

        Returns:
            Dictionary mapping each integer label to the set of
            vertices that share that label.
        """
        communities: Dict[int, Set[Hashable]] = {}
        for v, label in self._labels.items():
            if label not in communities:
                communities[label] = set()
            communities[label].add(v)
        return communities

    def get_label_vector(self, compress: bool = True) -> List[int]:
        """Return an n×1 label vector as a Python list.

        The i-th entry of the returned list corresponds to the i-th
        vertex in the original ``vertices`` iterable passed to
        ``__init__`` and contains the integer label (community) of that
        vertex.

        Args:
            compress: If True, the labels are remapped to a contiguous
                range 0..(k-1), where k is the number of distinct
                communities. If False, the raw integer labels produced
                by the algorithm are returned.

        Returns:
            List of length n where each element is an integer label
            indicating the community of the corresponding vertex.

        Raises:
            RuntimeError: If called before labels have been initialized
                (i.e., before calling :meth:`run`).
        """
        if not self._labels:
            raise RuntimeError("run() must be called before get_label_vector().")

        # Raw labels in the original vertex order
        raw_vec: List[int] = [self._labels[v] for v in self._vertex_list]

        if not compress:
            return raw_vec

        # Compress labels to contiguous 0..K-1
        unique_labels = sorted(set(raw_vec))
        remap: Dict[int, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
        return [remap[lbl] for lbl in raw_vec]
