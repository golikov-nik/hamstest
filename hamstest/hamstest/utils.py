from scipy.special import polygamma


def beta_mean_log(k: int, sample_size: int) -> float:
    return polygamma(0, k) - polygamma(0, sample_size + 1)


def beta_var_log(k: int, sample_size: int) -> float:
    return polygamma(1, k) - polygamma(1, sample_size + 1)
