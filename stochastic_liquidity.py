import numpy as np
from scipy.integrate import quad


def MC_stochastic_liquidity(
    S0,
    K,
    T,
    r,
    sigma,
    beta,
    alpha,
    theta,
    xi,
    rho1,
    rho2,
    rho3,
    L0,
    n_simulations,
    n_steps=100,
):
    # Monte Carlo simulation for the liquidity-adjusted option price
    dt = T / n_steps
    discount_factor = np.exp(-r * T)

    if type(K) is list:
        V = np.zeros((n_simulations, len(K)))
    else:
        V = np.zeros(n_simulations)

    # Cholesky decomposition for correlated Brownian motions
    cov_matrix = np.array([[1, rho1, rho2], [rho1, 1, rho3], [rho2, rho3, 1]])
    chol_matrix = np.linalg.cholesky(cov_matrix)

    St = np.zeros((n_simulations, n_steps + 1))
    Lt = np.zeros((n_simulations, n_steps + 1))

    St[:, 0] = S0
    Lt[:, 0] = L0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, (n_simulations, 3))
        dW = Z @ chol_matrix.T * np.sqrt(dt)

        dW1 = dW[:, 0]  # dB_S
        dW2 = dW[:, 1]  # dB_gamma
        dW3 = dW[:, 2]  # dB_L

        dSt = (r * dt + beta * Lt[:, t - 1] * dW2 + sigma * dW1) * St[:, t - 1]

        dLt = alpha * (theta - Lt[:, t - 1]) * dt + xi * dW3

        St[:, t] = St[:, t - 1] + dSt
        Lt[:, t] = Lt[:, t - 1] + dLt

    V = (
        np.array([np.maximum(St[:, -1] - k, 0) for k in K]).T
        if type(K) is list
        else np.maximum(St[:, -1] - K, 0)
    )
    price = np.mean(V, axis=0) * discount_factor

    return St, price


def characteristic_function_stochastic_liquidity(
    eta, r, sigma, beta, alpha, xi, theta, rho1, rho2, rho3, tau, liquidity, S
):
    j = complex(0, 1)

    delta1 = np.sqrt(
        (alpha - j * eta * rho3 * xi * beta) ** 2 + xi**2 * beta**2 * (j * eta + eta**2)
    )
    delta2 = (alpha - j * eta * rho3 * xi * beta) / delta1

    C_tau = (1 / (2 * xi**2)) * (
        (alpha - j * eta * rho3 * xi * beta)
        - delta1
        * (
            (np.sinh(delta1 * tau) + delta2 * np.cosh(delta1 * tau))
            / (np.cosh(delta1 * tau) + delta2 * np.sinh(delta1 * tau))
        )
    )

    delta4 = -(2 * alpha * theta + 2 * j * rho2 * xi * sigma * eta)
    delta3 = delta4 * (
        alpha - j * eta * rho3 * xi * beta
    ) + 2 * xi**2 * rho1 * sigma * beta * (j * eta + eta**2)

    B_tau = (1 / (2 * delta1 * xi**2)) * (
        (delta2 * delta3 - delta4 * delta1)
        / (np.cosh(delta1 * tau) + delta2 * np.sinh(delta1 * tau))
        + delta4 * delta1
    ) - (delta3 / (2 * delta1 * xi**2)) * (
        (np.sinh(delta1 * tau) + delta2 * np.cosh(delta1 * tau))
        / (np.cosh(delta1 * tau) + delta2 * np.sinh(delta1 * tau))
    )

    def integrand(t):
        return (
            (alpha * theta + j * rho2 * xi * sigma * eta)
            * (
                (1 / (2 * delta1 * xi**2))
                * (
                    (delta2 * delta3 - delta4 * delta1)
                    / (np.cosh(delta1 * t) + delta2 * np.sinh(delta1 * t))
                    + delta4 * delta1
                )
                - (delta3 / (2 * delta1 * xi**2))
                * (
                    (np.sinh(delta1 * t) + delta2 * np.cosh(delta1 * t))
                    / (np.cosh(delta1 * t) + delta2 * np.sinh(delta1 * t))
                )
            )  # the end of B_tau
            + 0.5
            * xi**2
            * (
                (1 / (2 * delta1 * xi**2))
                * (
                    (delta2 * delta3 - delta4 * delta1)
                    / (np.cosh(delta1 * t) + delta2 * np.sinh(delta1 * t))
                    + delta4 * delta1
                )
                - (delta3 / (2 * delta1 * xi**2))
                * (
                    (np.sinh(delta1 * t) + delta2 * np.cosh(delta1 * t))
                    / (np.cosh(delta1 * t) + delta2 * np.sinh(delta1 * t))
                )
            )
            ** 2  # the end of B_tau**2
            + xi**2
            * (
                (1 / (2 * xi**2))
                * (
                    (alpha - j * eta * rho3 * xi * beta)
                    - delta1
                    * (
                        (np.sinh(delta1 * t) + delta2 * np.cosh(delta1 * t))
                        / (np.cosh(delta1 * t) + delta2 * np.sinh(delta1 * t))
                    )
                )
            )
        )  # the end of C_tau

    A_tau = tau * (-0.5 * sigma**2 * (j * eta + eta**2) + j * r * eta)
    A2, _ = quad(integrand, 1e-8, tau, complex_func=True)
    value = A_tau + A2 + B_tau * liquidity + C_tau * liquidity**2 + j * eta * np.log(S)
    if value == np.nan:
        print(
            [eta, r, sigma, beta, alpha, xi, theta, rho1, rho2, rho3, tau, liquidity, S]
        )
        return 0
    return np.exp(value) if value.real <= 705 else 0


def price_closed_form_stochastic_liquidity(
    r, sigma, beta, alpha, xi, theta, rho1, rho2, rho3, tau, liquidity, S, K
):
    eta = np.linspace(1e-12, 100, 100)
    j = complex(0, 1)
    deta = eta[1] - eta[0]
    z = S * np.exp(r * tau)
    f1 = np.zeros_like(eta)
    f2 = np.zeros_like(eta)

    for i in range(len(eta)):
        f1[i] = np.real(
            characteristic_function_stochastic_liquidity(
                eta[i] - j,
                r,
                sigma,
                beta,
                alpha,
                xi,
                theta,
                rho1,
                rho2,
                rho3,
                tau,
                liquidity,
                S,
            )
            * np.exp(-j * eta[i] * np.log(K))
            / (j * eta[i])
        )
        f2[i] = np.real(
            characteristic_function_stochastic_liquidity(
                eta[i],
                r,
                sigma,
                beta,
                alpha,
                xi,
                theta,
                rho1,
                rho2,
                rho3,
                tau,
                liquidity,
                S,
            )
            * np.exp(-j * eta[i] * np.log(K))
            / (j * eta[i])
        )

    inte1 = 0
    inte2 = 0

    for i in range(1, len(eta) - 1):
        inte1 += f1[i] * deta
        inte2 += f2[i] * deta

    inte1 += (f1[0] + f1[-1]) * deta / 2
    inte2 += (f2[0] + f2[-1]) * deta / 2

    P1 = inte1 / np.pi / z + 0.5
    P2 = inte2 / np.pi + 0.5
    Price = S * P1 - K * np.exp(-r * tau) * P2

    return Price if Price > 0 else -100


def stochastic_liquidity_cost_function(
    x, cost_fn=lambda x, y: np.mean(np.abs(x - y)), filename="in_sample/2017-01-04.csv"
):
    print("hi")
    # Load the data from the file
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Determine the number of rows in the data
    n = data.shape[0]

    # Initialize the cost array
    calculated_price = np.zeros(n)

    # Loop over each data point to calculate the cost
    for i in range(n):
        calculated_price[i] = price_closed_form_stochastic_liquidity(
            data[i, 4],
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            data[i, 0],
            x[8],
            data[i, 1],
            data[i, 3],
        )

    # Calculate the final cost as the average of the individual costs
    final_cost = cost_fn(
        data[:, 2].astype(np.float64), calculated_price.astype(np.float64)
    )

    if final_cost == np.nan:
        print(f"{x=}")
        return 100_000

    return final_cost
