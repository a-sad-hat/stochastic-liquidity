import numpy as np
from scipy.optimize import differential_evolution

from stochastic_liquidity import (
    stochastic_liquidity_cost_function,
)

if __name__ == "__main__":
    # Define bounds for the parameters
    bounds = [
        (10**-6, 1),
        (10**-6, 5),
        (10**-6, 1),
        (10**-6, 1),
        (10**-6, 10),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (10**-6, 1),
    ]

    print("starting optimization")
    # Perform the optimization using differential evolution
    result = differential_evolution(
        stochastic_liquidity_cost_function,
        bounds,
        maxiter=3,
        disp=True,
        workers=-1,
        popsize=10,
    )
    print("optimization done")
    # Store the optimized parameters and the corresponding cost
    B = result.x
    cost = result.fun

    # Output the results
    print(f"Optimized parameters: {B}")
    print(f"Final cost: {cost}")

    # Save the results
    np.save("optimized_parameters_stochastic_liquidity.npy", B)
    np.save("final_cost+stochastic_liquidity.npy", cost)
