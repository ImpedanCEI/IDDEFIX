#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:33:41 2020

@author: sjoly
"""
import sys
import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution

class ProgressBarCallback:
    def __init__(self, max_generations):
        self.max_generations = max_generations
        self.current_generation = 0
        self.pbar = tqdm(total=max_generations, desc="Optimization Progress")

    def __call__(self, xk, convergence):
        self.current_generation += 1
        self.pbar.update(1)  # Update the progress bar
        if convergence < 1e-6:  # Convergence threshold
            self.pbar.close()
            return True  # Stop optimization early
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
        pbar = ProgressBarCallback(maxiter)
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

        # Need to be reworked to use the last population as the new initial population to speed up convergence
        """while ((result.message == 'Maximum number of iterations has been exceeded.') and (iteration_convergence)):
            warning = 'Increased number of iterations by 10% to reach convergence. \n'
            maxiter = int(1.1*maxiter)
            result = differential_evolution(minimization_function,parameterBounds, 
                                            popsize=popsize, tol=tol, maxiter=maxiter,
                                            mutation=mutation, recombination=crossover_rate, polish=False, 
                                            init='latinhypercube',
                                            callback=show_progress_bar,
                                            updating='deferred', workers=-1, #vectorized=vectorized
                                        )

        else:
            warning = ''
        """
        
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

        for i in range(maxiter):
            best, _ = solver.run(n_it=1)
            progress_bar_gui(1, np.max((tol / stop_criterion(solver), i / maxiter)))
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

        for i in range(maxiter):
            best, _ = solver.run(n_it=1)
            progress_bar_gui(1, np.max((tol / stop_criterion(solver), i / maxiter)))
            if stop_criterion(solver) < tol:
                break

        solution, message = best, "Convergence achieved" if i < maxiter else "Maximum iterations reached"

        return solution, message