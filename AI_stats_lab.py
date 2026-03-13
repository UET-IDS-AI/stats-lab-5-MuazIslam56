import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    PDF of exponential distribution
    f(x) = lam * exp(-lam*x)  for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Analytical probability P(a < X < b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Estimate probability using simulation
    """
    samples = np.random.exponential(scale=1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian PDF
    """
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    """

    mu_A = 40
    mu_B = 45
    sigma = 2

    P_A = 0.3
    P_B = 0.7

    fA = gaussian_pdf(time, mu_A, sigma)
    fB = gaussian_pdf(time, mu_B, sigma)

    return (fB * P_B) / (fA * P_A + fB * P_B)
