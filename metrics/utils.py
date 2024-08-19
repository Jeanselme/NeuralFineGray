from lifelines import KaplanMeierFitter

def estimate_ipcw(km):
    if isinstance(km, tuple):
        kmf = KaplanMeierFitter()
        e_train, t_train = km
        kmf.fit(t_train, e_train == 0)
    else: kmf = km
    return kmf