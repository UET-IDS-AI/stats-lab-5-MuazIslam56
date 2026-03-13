import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    x = np.array(x)
    pdf = lam * np.exp(-lam * x)
    pdf[x < 0] = 0
    return pdf


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)

    count = np.sum((samples > a) & (samples < b))

    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    return coefficient * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    mu_A = 40
    mu_B = 45
    sigma = 2     # variance = 4 → sigma = 2

    P_A = 0.3
    P_B = 0.7

    likelihood_A = gaussian_pdf(time, mu_A, sigma)
    likelihood_B = gaussian_pdf(time, mu_B, sigma)

    numerator = likelihood_B * P_B
    denominator = likelihood_A * P_A + likelihood_B * P_B

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    sigma = 2

    # priors
    P_A = 0.3
    P_B = 0.7

    # generate class labels
    classes = np.random.choice(["A", "B"], size=n, p=[P_A, P_B])

    times = np.zeros(n)

    for i in range(n):
        if classes[i] == "A":
            times[i] = np.random.normal(40, sigma)
        else:
            times[i] = np.random.normal(45, sigma)

    # choose swimmers with time close to given value
    tolerance = 0.1
    mask = np.abs(times - time) < tolerance

    selected_classes = classes[mask]

    if len(selected_classes) == 0:
        return 0

    B_count = np.sum(selected_classes == "B")

    return B_count / len(selected_classes)




