import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm


def MC_sotchastic_liquidity_and_regime_switching(
    r,
    sigma1,
    sigma2,
    lamda12,
    lamda21,
    beta,
    kappa,
    alpha,
    eta,
    rho,
    T,
    initial_liquidity,
    S0,
    K,
    N=150000,
    n_steps=100,
):
    theta = [1 / lamda12, 1 / lamda21]
    S = np.zeros(N)
    St = {f"{i}": [S0] for i in range(N)}
    liquidity = np.zeros(N)
    if type(K) is list:
        V = np.zeros((N, len(K)))
    else:
        V = np.zeros(N)

    for j in range(N):
        t = -theta[0] * np.log(np.random.rand())
        tau = [t]
        t0 = T - t
        k = 2
        while t0 > 0:
            if k % 2 == 0:
                theta0 = theta[1]
            else:
                theta0 = theta[0]
            t = -theta0 * np.log(np.random.rand())
            tau.append(t)
            t0 -= t
            k += 1

        tau[-1] = T - sum(tau) + tau[-1]

        S[j] = S0
        liquidity[j] = initial_liquidity

        for i in range(len(tau)):
            if i % 2 == 0:
                sigma = sigma1
            else:
                sigma = sigma2

            dt = tau[i] / n_steps
            W1 = np.random.randn(n_steps)
            W2 = np.random.randn(n_steps)
            W3 = rho * W2 + np.sqrt(1 - rho**2) * np.random.randn(n_steps)

            for z in range(n_steps):
                S[j] += (
                    r * S[j] * dt
                    + sigma * S[j] * np.sqrt(dt) * W1[z]
                    + beta * liquidity[j] * S[j] * np.sqrt(dt) * W2[z]
                )
                liquidity[j] += (
                    kappa * (alpha - liquidity[j]) * dt + eta * np.sqrt(dt) * W3[z]
                )
                St[f"{j}"].append(S[j])

            def payoff(K):
                return np.exp(-r * T) * max(S[j] - K, 0)

            V[j] = [payoff(k) for k in K] if type(K) is list else payoff(K)

    V1 = np.mean(V, axis=0)
    confint = 2.33 * np.std(V, axis=0) / np.sqrt(N)

    return V1, confint, St


def character_function_stochastic_liquidity_and_regime_switching(
    phi,
    r,
    sigma1,
    sigma2,
    lambda12,
    lambda21,
    beta,
    k,
    alpha,
    eta,
    rho,
    tau,
    liquidity,
    S,
):
    j = complex(0, 1)
    d = np.sqrt(
        (-2 * k + 2 * j * phi * rho * eta * beta) ** 2
        + 4 * eta**2 * (beta**2 * phi * j + beta**2 * phi**2)
    )
    g = (-2 * k + 2 * j * phi * rho * eta * beta - d) / (
        -2 * k + 2 * j * phi * rho * eta * beta + d
    )
    E = (
        (d - (-2 * k + 2 * j * phi * rho * eta * beta))
        / (4 * eta**2)
        * (1 - np.exp(d * tau))
        / (1 - g * np.exp(d * tau))
    )
    D = (
        -k
        * alpha
        * (d - (-2 * k + 2 * j * phi * rho * eta * beta))
        / (eta**2 * d)
        * (1 - np.exp(1 / 2 * d * tau)) ** 2
        / (1 - g * np.exp(d * tau))
    )

    def integrand(t):
        return 1 / 2 * eta**2 * (
            -k
            * alpha
            * (d - (-2 * k + 2 * j * phi * rho * eta * beta))
            / (eta**2 * d)
            * (1 - np.exp(1 / 2 * d * t)) ** 2
            / (1 - g * np.exp(d * t))
        ) ** 2 + k * alpha * (
            -k
            * alpha
            * (d - (-2 * k + 2 * j * phi * rho * eta * beta))
            / (eta**2 * d)
            * (1 - np.exp(1 / 2 * d * t)) ** 2
            / (1 - g * np.exp(d * t))
        )

    C1, _ = quad(integrand, 0, tau, complex_func=True)
    C2 = (
        1
        / 4
        * (
            (d - (-2 * k + 2 * j * phi * rho * eta * beta)) * tau
            - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g))
        )
        + j * phi * r * tau
    )
    A = np.array(
        [
            [
                -1 / 2 * (j * phi + phi**2) * sigma1**2 * tau - lambda12 * tau,
                lambda21 * tau,
            ],
            [
                lambda12 * tau,
                -1 / 2 * (j * phi + phi**2) * sigma2**2 * tau - lambda21 * tau,
            ],
        ]
    )
    L = expm(A)
    B = L[0, 0] + L[1, 0]
    ch = np.exp(C1 + C2 + D * liquidity + E * liquidity**2 + phi * j * np.log(S)) * B
    return ch


def price_closed_form_stochastic_liquidity_and_regime_switching(
    r,
    sigma1,
    sigma2,
    lambda12,
    lambda21,
    beta,
    k,
    alpha,
    eta,
    rho,
    tau,
    liquidity,
    S,
    K,
):
    phi = np.linspace(1e-12, 100, 1000)
    j = complex(0, 1)
    dphi = phi[1] - phi[0]
    z = S * np.exp(r * tau)
    f1 = np.zeros_like(phi)
    f2 = np.zeros_like(phi)

    for i in range(len(phi)):
        f1[i] = np.real(
            character_function_stochastic_liquidity_and_regime_switching(
                phi[i] - j,
                r,
                sigma1,
                sigma2,
                lambda12,
                lambda21,
                beta,
                k,
                alpha,
                eta,
                rho,
                tau,
                liquidity,
                S,
            )
            * np.exp(-j * phi[i] * np.log(K))
            / j
            / phi[i]
        )
        f2[i] = np.real(
            character_function_stochastic_liquidity_and_regime_switching(
                phi[i],
                r,
                sigma1,
                sigma2,
                lambda12,
                lambda21,
                beta,
                k,
                alpha,
                eta,
                rho,
                tau,
                liquidity,
                S,
            )
            * np.exp(-j * phi[i] * np.log(K))
            / j
            / phi[i]
        )

    inte1 = 0
    inte2 = 0

    for i in range(1, len(phi) - 1):
        inte1 += f1[i] * dphi
        inte2 += f2[i] * dphi

    inte1 += (f1[0] + f1[-1]) * dphi / 2
    inte2 += (f2[0] + f2[-1]) * dphi / 2

    P1 = inte1 / np.pi / z + 0.5
    P2 = inte2 / np.pi + 0.5
    Price = S * P1 - K * np.exp(-r * tau) * P2

    return Price


def stochastic_liquidity_and_regime_switching_cost_function(
    x, cost_fn=lambda x, y: np.mean((x - y) ** 2)
):
    print("hi_sl_rs")
    # Load the data from the file
    data = np.loadtxt("in_sample/2017-01-04.csv", delimiter=",", skiprows=1)
    # data = data[data[:, 0] == 1.534e-01]

    # Determine the number of rows in the data
    n = data.shape[0]

    # Initialize the cost array
    calculated_price = np.zeros(n)

    # Loop over each data point to calculate the cost
    for i in range(n):
        calculated_price[i] = (
            price_closed_form_stochastic_liquidity_and_regime_switching(
                data[i, 4],
                x[0],
                x[1],
                x[2],
                x[3],
                x[4],
                x[5],
                x[6],
                x[7],
                x[8],
                data[i, 0],
                x[9],
                data[i, 1],
                data[i, 3],
            )
        )

    # Calculate the final cost as the average of the individual costs
    final_cost = cost_fn(data[:, 2], calculated_price)

    return final_cost
