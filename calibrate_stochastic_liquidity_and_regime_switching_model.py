import numpy as np
from scipy.optimize import differential_evolution

from stochastic_liquidity_and_regime_switching import (
    stochastic_liquidity_and_regime_switching_cost_function,
)

if __name__ == "__main__":
    # Define bounds for the parameters
    bounds = [
        (10**-8, 1),
        (10**-8, 1),
        (10**-8, 10),
        (10**-8, 10),
        (10**-8, 5),
        (10**-8, 10),
        (10**-8, 1),
        (10**-8, 1),
        (-1, 1),
        (10**-8, 1),
    ]

    # Perform the optimization using differential evolution
    result = differential_evolution(
        stochastic_liquidity_and_regime_switching_cost_function,
        bounds,
        maxiter=3,
        disp=True,
        workers=-1,
        popsize=10,
    )

    # Store the optimized parameters and the corresponding cost
    B = result.x
    cost = result.fun

    # Output the results
    print(f"Optimized parameters: {B}")
    print(f"Final cost: {cost}")

    # Save the results
    np.save("optimized_parameters_stochastic_liquidity_and_regime_switching.npy", B)
    np.save("final_cost_stochastic_liquidity_and_regime_switching.npy", cost)
