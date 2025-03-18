"""
    Implementation of competing risk metrics.
"""
import numpy as np
from .utils import estimate_ipcw

epsilon = 1e-4

def brier_score(e_test, t_test, risk_predicted_test, times, t, km = None, competing_risk = 1):
    """
        Compute the corrected brier score for a given competing risk
        "Quantifying the predictive accuracty of time-to-event modes in the presence of competing risks" by Schoop et al.

        Args:
            e_test (array n): Indicator of event types (0 is censoring, all positive numbers encode competing events)
            t_test (array n): Time of observed event or censoring
            risk_predicted_test (array n * k): Matrix of predicted risk for the considered competing event for all test at all times
            times (array k): Times used for evaluating predictions
            t (float): Time at which to evaluate the brier score
            km (, optional): Kaplan Meier estimate or data to estimate censoring distribution. Defaults to None.
            competing_risk (int, optional): Competing risk for which to estimate calibration. Defaults to 1.

        Returns:
            float: Brier score for the considered competing risk evaluated at time t.
    """
    truth = (e_test == competing_risk) & (t_test <= t)
    index = np.argmin(np.abs(times - t))
    km = estimate_ipcw(km)

    if truth.sum() == 0:
        return np.nan, km

    if km is None:
        return ((truth - risk_predicted_test[:, index]) ** 2).mean(), km
    
    weights = np.zeros_like(e_test, dtype = float)
    weights[(t_test <= t) & (e_test != 0)] = 1. / np.clip(km.survival_function_at_times(t_test[(t_test <= t) & (e_test != 0)]), epsilon, None)
    weights[t_test > t] = 1. / np.clip(km.survival_function_at_times(t), epsilon, None)

    return (weights * (truth - risk_predicted_test[:, index]) ** 2).mean(), km

def integrated_brier_score(e_test, t_test, risk_predicted_test, times, t_eval = None, km = None, competing_risk = 1):
    """
        Integrated Brier score for competing risks
        Same as previous function but integrated over time, integrated at t_eval if given
    """
    km = estimate_ipcw(km)
    t_eval = times if t_eval is None else t_eval
    brier_scores = [brier_score(e_test, t_test, risk_predicted_test, times, t, km, competing_risk)[0] for t in t_eval]
    t_eval, brier_scores = t_eval[~np.isnan(brier_scores)], np.array(brier_scores)[~np.isnan(brier_scores)]

    if t_eval.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    return np.trapz(brier_scores, t_eval) / (t_eval[-1] - t_eval[0]), km