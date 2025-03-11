#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
@modified by: MaltheRaschke
"""
import numpy as np
from functools import partial
from scipy.optimize import minimize

from .solvers import Solvers
from .objectiveFunctions import ObjectiveFunctions as obj
from .resonatorFormulas import Impedances as imp
from .resonatorFormulas import Wakes as wak
from .utils import compute_fft

class EvolutionaryAlgorithm:
    def __init__(self, 
                 x_data, 
                 y_data,
                 N_resonators,
                 parameterBounds,
                 plane="longitudinal", 
                 fitFunction="impedance",
                 objectiveFunction = obj.sumOfSquaredError,
                 wake_length=None,
                 sigma=None,
                ):
        """
        Implements an evolutionary algorithm for fitting impedance models to data.

        This class optimizes a resonator-based impedance model to fit measured or 
        simulated impedance data using an evolutionary algorithm. It supports both 
        longitudinal and transverse impedance models with a variable number of resonators.

        Parameters
        ----------
        x_data : numpy.ndarray
            Stores the input x data: frequencies for impedance [Hz], or
            times for wake function and wake potential [s]
        y_data : numpy.ndarray
            Stores the input y data: impedance data [Ohm] or Wake function/potential [V/C/s]
        N_resonators : int
            Number of resonators in the model.
        parameterBounds : list of tuple
            List of parameter bounds for the optimization. Each resonator has three 
            parameters: Rs (shunt resistance), Q (quality factor), and fr (resonant frequency).
        fitFunction: str, optional
            Specify which fitFunction to use: ["impedance", "wake", "wake_potential"]
        plane : str, optional
            Type of impedance model, either `"longitudinal"` or `"transverse"`. 
            Default is `"longitudinal"`.
        objectiveFunction : callable, optional
            The objective function to minimize. Default is `obj.sumOfSquaredError`.
        wake_length : float, optional
            Length of the wake function in meters. Required for wake computations.
        sigma : float, optional

        Attributes
        ----------
        fitFunction : callable
            Partial function used to compute impedance based on the chosen model 
            (`imp.Resonator_longitudinal_imp`, `imp.n_Resonator_longitudinal_imp`, etc.).
        evolutionParameters : dict or None
            Stores parameters of the evolutionary optimization algorithm.
        minimizationParameters : numpy.ndarray or None
            Stores the best-fit parameters obtained from the optimization.
        time_data : numpy.ndarray
            Stores the input x data: times for wake function and wake potential [s]
        wake_data : numpy.ndarray
            Stores the input Wake function [V/C/s]
        wake_potential_data : numpy.ndarray
            Stores the input Wake potential [V/C/s]
        frequency_data : numpy.ndarray
            Stores the input x data: frequencies for impedance [Hz]
        impedance_data : numpy.ndarray
            Stores the input impedance data [Ohm]

        Notes
        -----
        - The `fitFunction` is assigned based on the `plane` type, the `fitFunction` mode, and the number of resonators.
        - The impedance and wake model is based on resonators and can be used for both 
        single-resonator and multi-resonator systems.
        - The optimization is performed using an evolutionary algorithm, with results 
        stored in `minimizationParameters`.

        Examples
        --------
        >>> import numpy as np
        >>> from evolutionary_algorithm import EvolutionaryAlgorithm
        >>> freq = np.linspace(1e9, 5e9, 100)  # Frequency range from 1 GHz to 5 GHz
        >>> Z = np.random.rand(100)  # Example impedance data
        >>> bounds = [(10, 1000), (1, 100), (1e9, 5e9)]  # Example bounds for Rs, Q, fr
        >>> algo = EvolutionaryAlgorithm(x_data=freq, y_data=Z, 
        ...                              N_resonators=1, parameterBounds=bounds)
        >>> print(algo.plane)
        'longitudinal'
        """

        self.x_data = x_data
        self.y_data = y_data

        self.N_resonators = N_resonators
        self.parameterBounds = parameterBounds
        self.objectiveFunction = objectiveFunction
        self.wake_length = wake_length
        self.plane = plane
        self.sigma = sigma

        self.time_data = None
        self.wake_data = None
        self.wake_potential_data = None
        self.frequency_data = None
        self.impedance_data = None
        self.evolutionParameters = None
        self.minimizationParameters = None
        
        if fitFunction == "wake" or fitFunction == "wake function":
            if plane == "longitudinal" and N_resonators > 1:
                self.fitFunction = wak.n_Resonator_longitudinal_wake
            elif plane == "transverse" and N_resonators > 1:
                self.fitFunction = wak.n_Resonator_transverse_wake
            elif plane == "longitudinal" and N_resonators == 1:
                self.fitFunction = wak.Resonator_longitudinal_wake
            elif plane == "transverse" and N_resonators == 1:
                self.fitFunction = wak.Resonator_transverse_wake
            else:
                raise Exception('Algorithm needs N_resonartors >= 1')
            self.time_data = x_data
            self.wake_data = y_data

        elif fitFunction == "wake potential":
            if self.sigma is None:
                print('[!] sigma not specified, using the default sigma=1e-10 s')
                self.sigma = 1e-10
                
            if plane == "longitudinal" and N_resonators > 1:
                self.fitFunction = partial(wak.n_Resonator_longitudinal_wake_potential, sigma=self.sigma)
            elif plane == "transverse" and N_resonators > 1:
                self.fitFunction = partial(wak.n_Resonator_transverse_wake_potential, sigma=self.sigma)
            elif plane == "longitudinal" and N_resonators == 1:
                self.fitFunction = partial(wak.Resonator_longitudinal_wake_potential, sigma=self.sigma)
            elif plane == "transverse" and N_resonators == 1:
                self.fitFunction = partial(wak.Resonator_transverse_wake_potential, sigma=self.sigma)
            else:
                raise Exception('Algorithm needs N_resonartors >= 1')
            self.time_data = x_data
            self.wake_potential_data = y_data
            
        else: #Default to "impedance"
            if wake_length is not None:
                print('[!] Using the partially decayed resonator formalism for impedance')
            else:
                print('[!] Using the fully decayed resonator formalism for impedance')

            if plane == "longitudinal" and N_resonators > 1:
                self.fitFunction = partial(imp.n_Resonator_longitudinal_imp, wake_length=wake_length)
            elif plane == "transverse" and N_resonators > 1:
                self.fitFunction = partial(imp.n_Resonator_transverse_imp, wake_length=wake_length)
            elif plane == "longitudinal" and N_resonators == 1:
                self.fitFunction = partial(imp.Resonator_longitudinal_imp, wake_length=wake_length)
            elif plane == "transverse" and N_resonators == 1:
                self.fitFunction = partial(imp.Resonator_transverse_imp, wake_length=wake_length)
            else:
                raise Exception('Algorithm needs N_resonartors >= 1') 
            self.frequency_data = x_data
            self.impedance_data = y_data
                
    def check_y_data(self):
        """
        Small function to avoid 0 frequency leading to zero division when using resonators.
        """
        mask = np.where(self.x_data > 0.)[0]
        self.x_data = self.x_data[mask]
        self.y_data = self.y_data[mask]
    

    def generate_Initial_Parameters(self, parameterBounds, objectiveFunction, fitFunction,
                                x_values_data, y_values_data,
                                maxiter=2000, popsize=150, 
                                mutation=(0.1, 0.5), crossover_rate=0.8,
                                tol=0.01,
                                solver='scipy',
                               ):    
        """
        Generates initial parameter guesses for a minimization algorithm.

        This function uses a differential evolution (DE) solver to find approximate
        solutions to a minimization problem, which can be used as initial guesses for
        more precise optimizers.

        Args:
            parameterBounds: A list of tuples representing the upper and lower bounds
                            for each parameter.
            objective_function: A function that calculates the cost given a set of
                            parameters and data. The signature should be
                            `objective_function(parameters, fit_function, x_data, y_data)`.
            fit_function: A function that calculates the fit between a model and data.
                        The signature should be `fit_function(parameters, x_data, y_data)`.
            x_values_data: The x-values of the data to fit.
            y_values_data: The y-values of the data to fit.
            maxiter: The maximum number of iterations for the DE solver.
            popsize: The population size for the DE algorithm.
            mutation: A tuple of two floats representing the mutation factors.
            crossover_rate: The crossover rate for the DE algorithm.
            tol: The tolerance for convergence.
            solver: The solver to use for differential evolution. Valid options are
                    "scipy", "pyfde", or "pyfde_jade". Defaults to "scipy".

        Returns:
            A tuple containing:
                - The estimated initial parameters found by the DE solver.
                - A message indicating the solver's status.
        """

        
        objective_function = partial(objectiveFunction, 
                                        fitFunction=fitFunction,
                                        x=x_values_data, 
                                        y=y_values_data
                                    )
        
        # Map solver names to functions
        solver_functions = {
            "scipy": Solvers.run_scipy_solver,
            "pyfde": Solvers.run_pyfde_solver,
            "pyfde_jade": Solvers.run_pyfde_jade_solver,
        }

        solver_function = solver_functions.get(solver)
        if solver == "pyfde_jade":
            mutation, crossover_rate = None, None
        
        if not solver_function:
            raise ValueError(f"Invalid solver name: {solver}")
            
        solution, message = solver_function(parameterBounds, 
                                            objective_function,
                                            maxiter=maxiter, 
                                            popsize=popsize, 
                                            mutation=mutation, 
                                            crossover_rate=crossover_rate,
                                            tol=tol)
        
        return solution, message
    
    def run_differential_evolution(self, 
                             maxiter=2000, 
                             popsize=15, 
                             mutation=(0.1, 0.5), 
                             crossover_rate=0.8, 
                             tol=0.01, 
                             #workers=-1, 
                             vectorized=False,
                             solver='scipy',
                             iteration_convergence=False, debug=False):
        evolutionParameters, warning = self.generate_Initial_Parameters(self.parameterBounds, 
                                                           self.objectiveFunction, 
                                                           self.fitFunction, 
                                                           self.x_data, 
                                                           self.y_data, 
                                                           maxiter=maxiter, 
                                                           popsize=popsize, 
                                                           mutation=mutation, 
                                                           crossover_rate=crossover_rate,
                                                           tol=tol,
                                                           solver=solver
                                                           #workers=workers, vectorized=vectorized,
                                                           #iteration_convergence=iteration_convergence
                                                                )

        self.evolutionParameters = evolutionParameters
        self.warning = warning
        self.display_resonator_parameters(self.evolutionParameters)
            
    def run_minimization_algorithm(self, margin=[0.1, 0.1, 0.1], method='Nelder-Mead'):
        """
        Runs a minimization algorithm to refine resonance parameters.

        This function refines the parameters obtained from the Differential Evolution (DE) 
        algorithm by using a local optimization method. If the DE algorithm has not been run, 
        it directly minimizes the objective function using initial parameter bounds.

        Each parameter is allowed to vary within a specified margin, where:
        - Rs values use `margin[0]`
        - Q values use `margin[1]`
        - fres values use `margin[2]`

        Parameters
        ----------
        margin : float or list of float, optional
            A list of three values specifying the relative margins for Rs, Q, and fres.
            Each parameter is allowed to vary by ±(margin * value). Default is [0.1, 0.1, 0.1].
        method : str, optional
            Optimization method for `scipy.optimize.minimize`. Default is 'Nelder-Mead'.

        Notes
        -----
        - The optimization is constrained within `minimizationBounds`, which are computed 
        using `margin` and the current `evolutionParameters`.
        - If the DE algorithm has not been run, the function initializes parameters using 
        `self.parameterBounds` and minimizes the objective function.
        - The minimization results are stored in `self.minimizationParameters`.
        - Calls `self.display_resonator_parameters()` to display the refined parameters.

        Returns
        -------
        None
            The refined parameters are stored in `self.minimizationParameters`.
        """

        print('Method for minimization : '+method)
        objective_function = partial(self.objectiveFunction, fitFunction=self.fitFunction,
                                     x=self.x_data, y=self.y_data)
        if type(margin) is float:
            margin = [margin] * 3

        if self.evolutionParameters is not None:
            # Apply different margins based on parameter type (Rs, Q, fres)
            minimizationBounds = [
                sorted(((1 - margin[i % 3]) * p, (1 + margin[i % 3]) * p))  
                for i, p in enumerate(self.evolutionParameters)
                ]
            minimizationParameters = minimize(objective_function, 
                                              x0=self.evolutionParameters,
                                              bounds=minimizationBounds,
                                              tol=1, #empiric value, documentation is cryptic
                                              method=method, 
                                              options={'maxiter': self.N_resonators * 1000,
                                                       'maxfev': self.N_resonators * 1000,
                                                       'disp': False,
                                                       'adaptive': True}
                                             )
        else:
            print('Differential Evolution algorithm not run, minimization only')
            minimizationParameters = minimize(objective_function, 
                                              x0=np.mean(self.parameterBounds, axis=1),
                                              bounds=self.parameterBounds, 
                                              method=method, 
                                              tol=1,
                                              options={'maxiter': self.N_resonators * 5000,
                                                       'maxfev': self.N_resonators * 5000,
                                                       'disp': False,
                                                       'adaptive': True}
                                             ) 
        self.minimizationParameters = minimizationParameters.x
        self.display_resonator_parameters(self.minimizationParameters)
    
    def display_resonator_parameters(self, params=None, to_markdown=False):
        """
        Displays resonance parameters in a formatted table using ASCII characters.

        Args:
            solution: A NumPy array of resonator parameters, typically shaped (n_resonators, 3).
        """
        header_format = "{:^10}|{:^24}|{:^18}|{:^18}"
        data_format = "{:^10d}|{:^24.2e}|{:^18.2f}|{:^18.3e}"
        if to_markdown:
            print("\n")
            print("| Resonator | Rs [Ohm/m or Ohm] | Q | fres [Hz] |")
            print("|-----------|------------------|---|-----------|")
            for i, parameters in enumerate(params.reshape(-1, 3)):
                print(f"| {i + 1} | {parameters[0]:.6g} | {parameters[1]:.6g} | {parameters[2]:.6g} |")
        else:
            print("\n")
            print("-" * 70)

            # Print header
            print(header_format.format("Resonator", "Rs [Ohm/m or Ohm]", "Q", "fres [Hz]"))
            print("-" * 70)

            # Print data
            for i, parameters in enumerate(params.reshape(-1,3)):
                print(data_format.format(i + 1, *parameters))

            print("-" * 70)

    def get_wake(self, time_data=None, use_minimization=True):

        # Check for time data 
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula
        if self.plane == "longitudinal" and self.N_resonators > 1:
            wake_data = wak.n_Resonator_longitudinal_wake(time_data, pars)
        elif self.plane == "transverse" and self.N_resonators > 1:
            wake_data = wak.n_Resonator_transverse_wake(time_data, pars)
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            wake_data = wak.Resonator_longitudinal_wake(time_data, pars)
        elif self.plane == "transverse" and self.N_resonators == 1:
            wake_data = wak.Resonator_transverse_wake(time_data, pars)

        if self.wake_data is None:
            self.wake_data = wake_data

        return wake_data
    
    def get_wake_potential(self, time_data=None, sigma=1e-9, use_minimization=True):

        # Check for time data 
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula
        if self.plane == "longitudinal" and self.N_resonators > 1:
            wake_potential_data = wak.n_Resonator_longitudinal_wake_potential(time_data, pars, sigma=sigma)
        elif self.plane == "transverse" and self.N_resonators > 1:
            wake_potential_data = wak.n_Resonator_transverse_wake_potential(time_data, pars, sigma=sigma)
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            wake_potential_data = wak.Resonator_longitudinal_wake_potential(time_data, pars, sigma=sigma)
        elif self.plane == "transverse" and self.N_resonators == 1:
            wake_potential_data = wak.Resonator_transverse_wake_potential(time_data, pars, sigma=sigma)

        return wake_potential_data
    
    def get_impedance_from_fitFunction(self, frequency_data=None, use_minimization=True):
        # Check for frequency data 
        if frequency_data is None:
            if self.frequency_data is None:
                raise AttributeError("Provide frequency data array")
            frequency_data = self.frequency_data
        else:
            if self.frequency_data is None:
                self.frequency_data = frequency_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        impedance_data = self.fitFunction(frequency_data, pars)
        
        return impedance_data

    def get_impedance(self, frequency_data=None,
                      use_minimization=True, wakelength=None):
        # Check for frequency data 
        if frequency_data is None:
            if self.frequency_data is None:
                raise AttributeError("Provide frequency data array")
            frequency_data = self.frequency_data
        else:
            if self.frequency_data is None:
                self.frequency_data = frequency_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula
        if self.plane == "longitudinal" and self.N_resonators > 1:
            impedance_data = imp.n_Resonator_longitudinal_imp(frequency_data, pars, wakelength)
        elif self.plane == "transverse" and self.N_resonators > 1:
            impedance_data = imp.n_Resonator_transverse_imp(frequency_data, pars, wakelength)
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            impedance_data = imp.Resonator_longitudinal_imp(frequency_data, pars, wakelength)
        elif self.plane == "transverse" and self.N_resonators == 1:
            impedance_data = imp.Resonator_transverse_imp(frequency_data, pars, wakelength)
        
        return impedance_data
    
    def get_impedance_from_fft(self, time_data=None, wake_data=None, 
                               fmax=3e9, samples=1001):
        # Check for time data 
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data
        
        wake_data = self.get_wake(self.time_data)

        f, Z = compute_fft(data_time=time_data,
                           data_wake=wake_data,
                           fmax=fmax, 
                           samples=samples)
        
        if self.plane == 'transverse':
            Z *= -1j
        
        return f, Z

    def compute_fft(self, data_time=None, data_wake=None, fmax=3e9, samples=1001):
        # Check for time data - not override self
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data

        # Check for wake data - not override self
        if wake_data is None:
            if self.wake_data is None:
                raise AttributeError("Provide wake data array")
            wake_data = self.wake_data

        compute_fft(data_time, data_wake, fmax, samples)

    def get_extrapolated_wake(self, new_end_time=None, dt=None, 
                              time_data=None, use_minimization=True):

        # Check for time data 
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide `time_data` array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        if new_end_time is None:
            raise Exception('Provide `new_end_time` to extrapolate')

        if dt is None:
            dt = np.min(time_data[1:]-time_data[:-1])

        ext_time_data = np.concatenate((time_data[:-1], 
                                       np.arange(time_data[-1], new_end_time, dt)))
        
        ext_wake_data = self.get_wake(ext_time_data, use_minimization)

        return ext_time_data, ext_wake_data   
    