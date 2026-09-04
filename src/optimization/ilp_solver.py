"""Global Integer Linear Programming (ILP) Lineage Solver using SCIP (pyscipopt)."""

from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import pyscipopt
import tracksdata

from src.graph.candidate_graph import CandidateEdge, CandidateGraph, CandidateNode
from src.graph.graph_export import solution_edges_to_tracksdata

logger = logging.getLogger(__name__)


@dataclass
class ILPSolution:
    """Optimal cell lineage solution extracted from CandidateGraph."""

    selected_node_ids: List[int]
    selected_edge_ids: List[int]
    dividing_node_ids: List[int]
    appearing_node_ids: List[int]
    disappearing_node_ids: List[int]
    objective_value: float
    solve_time_sec: float
    status: str
    num_tracks: int = 0

    def to_tracksdata(self, graph: CandidateGraph) -> tracksdata.graph.IndexedRXGraph:
        """Export solved lineages to tracksdata IndexedRXGraph."""
        return solution_edges_to_tracksdata(graph, self.selected_edge_ids)


class LineageILPSolver:
    """Global Integer Linear Programming solver for cell lineage tracking with conservation of flow."""

    def __init__(
        self,
        weight_edge: float = 1.0,
        weight_node: float = 0.5,
        weight_division: float = 1.0,
        cost_appear: float = 3.0,
        cost_disappear: float = 3.0,
        prob_prune_threshold: float = 0.05,
        time_limit_sec: float = 120.0,
        quiet: bool = True,
    ):
        """
        Args:
            weight_edge: Weight multiplier on transition edge costs.
            weight_node: Weight multiplier on node selection costs.
            weight_division: Weight multiplier on mitosis costs.
            cost_appear: Cost for track initiation (appearance).
            cost_disappear: Cost for track termination (disappearance).
            prob_prune_threshold: Discard candidate edges below this probability to reduce problem size.
            time_limit_sec: Maximum SCIP solver execution time in seconds.
            quiet: If True, suppress solver console output.
        """
        self.weight_edge = weight_edge
        self.weight_node = weight_node
        self.weight_division = weight_division
        self.cost_appear = cost_appear
        self.cost_disappear = cost_disappear
        self.prob_prune_threshold = prob_prune_threshold
        self.time_limit_sec = time_limit_sec
        self.quiet = quiet

    def solve(self, graph: CandidateGraph) -> ILPSolution:
        """Build and solve the ILP model on the candidate tracking graph.

        Args:
            graph: CandidateGraph populated with nodes, edges, and deep learning costs.

        Returns:
            ILPSolution with optimal active lineage edges, nodes, and divisions.
        """
        start_time = time.time()

        if graph.num_nodes == 0:
            return ILPSolution(
                selected_node_ids=[],
                selected_edge_ids=[],
                dividing_node_ids=[],
                appearing_node_ids=[],
                disappearing_node_ids=[],
                objective_value=0.0,
                solve_time_sec=0.0,
                status="empty_graph",
                num_tracks=0,
            )

        model = pyscipopt.Model("CellLineageILP")
        if self.quiet:
            model.hideOutput()

        model.setParam("limits/time", self.time_limit_sec)

        # Filtered edges (pruning near-zero probability candidates)
        active_candidate_edges: Dict[int, CandidateEdge] = {
            eid: e for eid, e in graph.edges.items() if e.probability >= self.prob_prune_threshold
        }

        # Sub-adjacencies for active candidate edges
        node_out_edges: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
        node_in_edges: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}

        for eid, e in active_candidate_edges.items():
            node_out_edges[e.source_id].append(eid)
            node_in_edges[e.target_id].append(eid)

        # ----------------------------------------------------
        # 1. Decision Variables
        # ----------------------------------------------------
        # x_e: binary selection for candidate edge e
        var_x: Dict[int, pyscipopt.Variable] = {}
        for eid in active_candidate_edges:
            var_x[eid] = model.addVar(name=f"x_{eid}", vtype="B")

        # y_u: binary selection for candidate cell node u
        var_y: Dict[int, pyscipopt.Variable] = {}
        # a_u: appearance indicator
        var_a: Dict[int, pyscipopt.Variable] = {}
        # d_u: disappearance indicator
        var_d: Dict[int, pyscipopt.Variable] = {}
        # m_u: division (mitosis) indicator
        var_m: Dict[int, pyscipopt.Variable] = {}

        for nid in graph.nodes:
            var_y[nid] = model.addVar(name=f"y_{nid}", vtype="B")
            var_a[nid] = model.addVar(name=f"a_{nid}", vtype="B")
            var_d[nid] = model.addVar(name=f"d_{nid}", vtype="B")
            var_m[nid] = model.addVar(name=f"m_{nid}", vtype="B")

        # ----------------------------------------------------
        # 2. Conservation of Flow Constraints
        # ----------------------------------------------------
        for nid, node in graph.nodes.items():
            in_eids = node_in_edges[nid]
            out_eids = node_out_edges[nid]

            # Inflow conservation: a_u + sum(incoming x_e) == y_u
            in_expr = var_a[nid] + pyscipopt.quicksum(var_x[eid] for eid in in_eids)
            model.addCons(in_expr == var_y[nid], name=f"flow_in_{nid}")

            # Outflow conservation: d_u + sum(outgoing x_e) == y_u + m_u
            out_expr = var_d[nid] + pyscipopt.quicksum(var_x[eid] for eid in out_eids)
            model.addCons(out_expr == var_y[nid] + var_m[nid], name=f"flow_out_{nid}")

            # Mitosis conditions:
            # - Division can only occur if node is selected: m_u <= y_u
            model.addCons(var_m[nid] <= var_y[nid], name=f"div_active_{nid}")

            # - Division requires at least 2 outgoing edges: sum(outgoing x_e) >= 2 * m_u
            model.addCons(
                pyscipopt.quicksum(var_x[eid] for eid in out_eids) >= 2 * var_m[nid],
                name=f"div_two_daughters_{nid}",
            )

            # - A dividing cell cannot disappear: d_u + m_u <= y_u
            model.addCons(var_d[nid] + var_m[nid] <= var_y[nid], name=f"div_no_disappear_{nid}")

            # - Out-degree capacity: at most 2 outgoing edges per cell
            model.addCons(
                pyscipopt.quicksum(var_x[eid] for eid in out_eids) <= 2 * var_y[nid],
                name=f"max_out_{nid}",
            )

            # - In-degree capacity: at most 1 incoming edge per cell
            model.addCons(
                pyscipopt.quicksum(var_x[eid] for eid in in_eids) <= var_y[nid],
                name=f"max_in_{nid}",
            )

        # ----------------------------------------------------
        # 3. Objective Function
        # ----------------------------------------------------
        obj_terms = []

        # Edge transition costs
        for eid, e in active_candidate_edges.items():
            edge_cost = self.weight_edge * e.cost
            obj_terms.append(edge_cost * var_x[eid])

        # Node selection, appearance, disappearance, and mitosis costs
        for nid, node in graph.nodes.items():
            node_cost = self.weight_node * node.cost_select
            div_cost = self.weight_division * node.cost_division
            appear_cost = self.cost_appear
            disappear_cost = self.cost_disappear

            obj_terms.append(node_cost * var_y[nid])
            obj_terms.append(appear_cost * var_a[nid])
            obj_terms.append(disappear_cost * var_d[nid])
            obj_terms.append(div_cost * var_m[nid])

        model.setObjective(pyscipopt.quicksum(obj_terms), "minimize")

        # ----------------------------------------------------
        # 4. Optimize
        # ----------------------------------------------------
        model.optimize()

        solve_time = time.time() - start_time
        status = model.getStatus()

        selected_edges: List[int] = []
        selected_nodes: List[int] = []
        dividing_nodes: List[int] = []
        appearing_nodes: List[int] = []
        disappearing_nodes: List[int] = []

        if status in ("optimal", "timelimit", "bndlim", "userinterrupt") and model.getNSols() > 0:
            sol = model.getBestSol()

            for eid, v in var_x.items():
                if model.getSolVal(sol, v) > 0.5:
                    selected_edges.append(eid)

            for nid, v in var_y.items():
                if model.getSolVal(sol, v) > 0.5:
                    selected_nodes.append(nid)

            for nid, v in var_m.items():
                if model.getSolVal(sol, v) > 0.5:
                    dividing_nodes.append(nid)

            for nid, v in var_a.items():
                if model.getSolVal(sol, v) > 0.5:
                    appearing_nodes.append(nid)

            for nid, v in var_d.items():
                if model.getSolVal(sol, v) > 0.5:
                    disappearing_nodes.append(nid)

            obj_val = float(model.getSolObjVal(sol))
        else:
            logger.warning(f"SCIP did not find a feasible solution. Status: {status}")
            obj_val = float("inf")

        return ILPSolution(
            selected_node_ids=selected_nodes,
            selected_edge_ids=selected_edges,
            dividing_node_ids=dividing_nodes,
            appearing_node_ids=appearing_nodes,
            disappearing_node_ids=disappearing_nodes,
            objective_value=obj_val,
            solve_time_sec=round(solve_time, 4),
            status=status,
            num_tracks=len(appearing_nodes),
        )
