#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:33:41 2020

@author: sjoly
@contributor: edelafue, babreufig
"""
import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution

class ProgressBarCallback:
    def __init__(self, max_generations, desc="Optimization"):
        self.max_generations = max_generations
        self.current_generation = 0
        self.pbar = tqdm(total=max_generations, desc=desc, unit="gen")

    def __call__(self, *args):

        # --- scipy differential_evolution: (xk, convergence) ---
        if len(args) == 2 and not hasattr(args[0], "evaluator"):
            xk, convergence = args
            self.current_generation += 1
            self.pbar.update(1)
            self.pbar.set_postfix({
                "conv": f"{(100*(convergence)):6.1f} %"
            })
            # optional early stopping
            if convergence >= 1.0:
                self.pbar.close()
                return True
        
        # --- pymoo minimize: (algorithm) ---
        elif len(args) == 1:
            algorithm = args[0]
            self.current_generation += 1
            self.pbar.update(1)

            # compute convergence metric
            F = np.atleast_1d(algorithm.pop.get("F"))
            gap = float(np.mean(F) - np.min(F))
            self._gap0 = getattr(self, "_gap0", gap if gap > 0 else 1.0)
            convergence = float(np.clip(1.0 - gap / (self._gap0 + 1e-12), 0.0, 1.0))

            self.pbar.set_postfix({
                "conv": f"{(100*(convergence)):6.1f} %"
            })
            # optional early stopping
            if convergence >= 1.0 and self.current_generation > 1:
                return True

    def close(self):
        self.pbar.close()

def stop_criterion(solver):
    '''
    Based on the criterion used in scipy.optimize.differential_evolution
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    Check that the ratio of the spread of the population fitness compared to its average.
    In other words, if most of the population individuals converge to the same solution indicating
    an optimal solution has been found.
    '''
    population_cost = np.vstack(solver)[:,1]
    population_mean, population_std = np.mean(population_cost), np.std(population_cost)
    criterion = population_std / np.abs(population_mean)
    return criterion

class Solvers:
    def run_scipy_solver(parameterBounds,
                        minimization_function,
                        maxiter=2000,
                        popsize=150,
                        mutation=(0.1, 0.5),
                        crossover_rate=0.8,
                        tol=0.01,
                        **kwargs):
        """
        Runs the SciPy differential_evolution solver to minimize a given function.

        All the arguments are detailed on this page :
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
        Setting workers= -1 means all CPUs available will be used for the computation.
        Default parameters used for the DE algorithm taken from https://www.mdpi.com/2227-7390/9/4/427

        Args:
            parameterBounds: A list of tuples representing the upper and lower bounds for each parameter.
            minimization_function: The function to be minimized.
            maxiter: The maximum number of iterations to run the solver for.
            popsize: The population size for the differential evolution algorithm.
            mutation: A tuple of two floats representing the mutation factors.
            crossover_rate: The crossover rate for the differential evolution algorithm.
            tol: The tolerance for convergence.

        Returns:
            A tuple containing:
                - The solution found by the solver.
                - A message indicating the solver's status.
        """
        pbar = ProgressBarCallback(maxiter, desc='Differential Evolution')
        result = differential_evolution(minimization_function,
                                        parameterBounds,
                                        popsize=popsize,
                                        tol=tol,
                                        maxiter=maxiter,
                                        mutation=mutation,
                                        recombination=crossover_rate,
                                        polish=False,
                                        init='latinhypercube',
                                        strategy='rand1bin',
                                        callback=pbar,
                                        updating='deferred',
                                        workers=-1,
                                        **kwargs,
                                    )
        pbar.close()

        solution, message = result.x, result.message

        return solution, message

    def run_pyfde_solver(parameterBounds,
                        minimization_function,
                        maxiter=2000,
                        popsize=150,
                        mutation=(0.45),
                        crossover_rate=0.8,
                        tol=0.01,
                        **kwargs):
        """
        Runs the pyfde ClassicDE solver to minimize a given function.

        Args:
            parameterBounds: A list of tuples representing the bounds for each parameter.
            minimization_function: The function to be minimized.
            maxiter: The maximum number of iterations to run the solver for.
            popsize: The population size for the differential evolution algorithm.
            mutation: A tuple of two floats representing the mutation factors.
            crossover_rate: The crossover rate for the differential evolution algorithm.
            tol: The tolerance for convergence.

        Returns:
            A tuple containing:
                - The solution found by the solver.
                - A message indicating the solver's status.
        """
        try:
            from pyfde import ClassicDE
        except:
            raise ImportError("Please install the pyfde package to use the pyfde solvers.")

        solver = ClassicDE(
            minimization_function,
            n_dim=len(parameterBounds),
            n_pop=popsize * len(parameterBounds),
            limits=parameterBounds,
            minimize=True,
        )
        solver.cr, solver.f = crossover_rate, np.mean(np.atleast_1d(mutation))

        for i in tqdm(range(maxiter)):
            best, _ = solver.run(n_it=1)
            if stop_criterion(solver) < tol:
                break

        solution, message = best, "Convergence achieved" if i < maxiter else "Maximum iterations reached"

        return solution, message

    def run_pyfde_jade_solver(parameterBounds,
                            minimization_function,
                            maxiter=2000,
                            popsize=150,
                            tol=0.01,
                            **kwargs):
        """
        Runs the pyfde JADE solver to minimize a given function.

        Args:
            parameterBounds: A list of tuples representing the bounds for each parameter.
            minimization_function: The function to be minimized.
            maxiter: The maximum number of iterations to run the solver for.
            popsize: The population size for the differential evolution algorithm.
            tol: The tolerance for convergence.

        Returns:
            A tuple containing:
                - The solution found by the solver.
                - A message indicating the solver's status.
        """

        try:
            from pyfde import JADE
        except:
            raise ImportError("Please install the pyfde package to use the pyfde solvers.")

        solver = JADE(
            minimization_function,
            n_dim=len(parameterBounds),
            n_pop=popsize * len(parameterBounds),
            limits=parameterBounds,
            minimize=True,
        )

        for i in tqdm(range(maxiter)):
            best, _ = solver.run(n_it=1)
            if stop_criterion(solver) < tol:
                break

        solution, message = best, "Convergence achieved" if i < maxiter else "Maximum iterations reached"

        return solution, message

    def run_pymoo_cmaes_solver(parameterBounds,
                            minimization_function,
                            sigma=0.1,
                            maxiter=None, # default: 100 + 150 * (N+3)**2 // popsize**0.5
                            popsize=None, # defaul: 4 + int(3 * np.log(len(parameterBounds)))
                            verbose=False,
                            **kwargs):
        """
        Runs the pymoo CMAES solver to minimize a given function.

        Args:
            parameterBounds: A list of tuples representing the bounds for each parameter.
            minimization_function: The function to be minimized.
            maxiter: The maximum number of iterations to run the solver for.
            popsize: The population size for the differential evolution algorithm.
            tol: The tolerance for convergence.

        Returns:
            A tuple containing:
                - The solution found by the solver.
                - A message indicating the solver's status.
        """
        try:
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            from pymoo.termination import get_termination
        except Exception:
            raise ImportError('''Please install the pymoo package to use the CMA-ES solver:
                            >>> pip install pymoo
                            ''')

        class OptimizationProblem(Problem):
            def __init__(self, objective_function, n_var, n_obj, xl, xu):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                self.objective_function = objective_function

            def _evaluate(self, x, out):
                out["F"] = [self.objective_function(xi) for xi in x]
                
        problem = OptimizationProblem(
            objective_function=minimization_function,
            n_var=len(parameterBounds),
            n_obj=1,
            xl=[bound[0] for bound in parameterBounds],
            xu=[bound[1] for bound in parameterBounds],
        )

        # Calculate mean of parameter bounds as starting point
        x0 = np.mean(parameterBounds, axis=1)

        solver = CMAES(
            x0=x0,
            sigma=sigma,
            popsize=popsize,
            maxfevals=maxiter,
            seed=42,
            restarts=3,
            restart_from_best=True,
            verb_time=verbose,
            **kwargs,
        )

        if not verbose:
            cb = ProgressBarCallback(maxiter, desc="CMA-ES evolution")  
        else: 
            cb = None

        res = minimize(problem, solver, 
                       seed=42, 
                       callback=cb,
                       save_history=True,)

        if not verbose: 
            cb.close()

        solution = res.X
        message = "Convergence achieved" if res.algorithm.n_gen < maxiter else "Maximum iterations reached"

        return solution, message, res
