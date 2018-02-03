from .base_segmenter import SegmenterFromCosts
from .... import Configuration

import numpy as np
import nifty.graph.opt.multicut as nmc


def transform_probabilities_to_costs(probabilities, beta=.5, edge_sizes=None):
    p_min = 0.001
    p_max = 1. - p_min
    costs = (p_max - p_min) * probabilities + p_min
    # probabilities to costs, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)
    # weight the costs with edge sizes, if they are given
    if edge_sizes is not None:
        assert len(edge_sizes) == len(costs)
        w = edge_sizes / edge_sizes.max()
        costs *= w
    return costs


class Multicut(SegmenterFromCosts):
    solvers = ["greedy-additive", "kernighan-lin", "fusion-moves", "ilp"]

    def __init__(self, solver, **solver_options):
        super(SegmenterFromCosts, self).__init__()
        assert solver in self.solvers
        if solver == "ilp":
            assert self._have_ilp()
        self.solver = solver
        self.solver_options = solver_options

    @staticmethod
    def _have_ilp():
        return Configuration.WITH_CPLEX or Configuration.WITH_GUROBI or Configuration.WITH_GLPK

    def _get_fusion_moves(self, objective):
        # get the proposal generator
        gen = self.solver_options.get("proposal_generator", None)
        if gen == "watershed" or gen is None:
            sigma = self.solver_options.get("sigma", 1.)
            n_seeds = self.solver_options.get("n_seeds", 1.)
            generator = objective.watershedCcProposals(sigma, n_seeds)
        elif gen == "interface_flipper":
            generator = objective.interFaceFlipperCcProposals()
        elif gen == "random_node_color":
            n_colors = self.solver_options.get("n_colors", 2)
            generator = objective.randomNodeColorProposals(n_colors)
        else:
            raise RuntimeError("Invalid fusion move proposal generator")

        # get the solver backend
        backend = self.solver_options.get("backend", "kernighan-lin")

        if backend == "greedy-additive":
            solver_backend = objective.greedyAdditiveFactory()
        elif backend == "kernighan-lin":
            solver_backend = objective.kernighanLinFactory(warmstartGreedy=True)
        elif backend == "ilp":
            assert self._have_ilp()
            solver_backend = objective.multicutIlpFactory()
        else:
            raise RuntimeError("Invalid fusion move backend")

        # get the other options
        n_threads = self.solver_options.get("n_threads", 1)
        n_iter = self.solver_options.get("n_iter", 100)
        n_stop = self.solver_options.get("n_stop", 10)

        fm = objective.ccFusionMoveBasedFactory(proposalGenerator=generator,
                                                numberOfThreads=n_threads,
                                                numberOfIterations=n_iter,
                                                stopIfNoImprovement=n_stop,
                                                fusionMove=solver_backend)
        # TODO need to set this if we use verbosity
        # chain solvers for warmstarting
        warmstart_greedy = self.solver_options.get("warmstart_greedy", True)
        warmstart_kl = self.solver_options.get("warmstart_kl", True)
        if warmstart_greedy and warmstart_kl:
            ga_factory = objective.greedyAdditiveFactory()
            kl_factory = objective.kernighanLinFactory()
            fm = objective.chainedSolversFactory([ga_factory, kl_factory, fm])
        elif warmstart_greedy:
            ga_factory = objective.greedyAdditiveFactory()
            fm = objective.chainedSolversFactory([ga_factory, fm])
        elif warmstart_kl:
            kl_factory = objective.kernighanLinFactory()
            fm = objective.chainedSolversFactory([kl_factory, fm])

        return fm.create(objective)

    # TODO kwarg for verbosity
    # TODO support logging visitor
    def _segmentation_impl(self, graph, costs, time_limit=None, **kwargs):
        objective = nmc.multicutObjective(graph, costs)

        if self.solver == 'greedy-additive':
            solver_impl = objective.greedyAdditiveFactory().create(objective)
        elif self.solver == 'kernighan-lin':
            warmstart = self.solver_options.get('warmstart_greedy', True)
            # TODO need to set this if we use verbosity
            # greedyVisitNth = kwargs.pop('greedyVisitNth', 100)
            solver_impl = objective.kernighanLinFactory(warmstartGreedy=warmstart).create(objective)
        elif self.solver == 'fusion-moves':
            solver_impl = self._get_fusion_moves(objective)
        elif self.solver == 'ilp':
            ilp_backend = self.solver_options.get("ilp_backend", None)
            solver_impl = objective.multicutIlpFactory(ilpSolver=ilp_backend).create(objective)

        # TODO this needs to change once we suport verbosity / logging
        if time_limit is None:
            node_labels = solver_impl.optimize()
        else:
            visitor = objective.verboseVisitor(visitNth=100000000, timeLimitSolver=time_limit)
            node_labels = solver_impl.optimize(visitor=visitor)

        return node_labels
