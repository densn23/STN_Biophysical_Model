NEURON {
    POINT_PROCESS DepExp2Syn
    RANGE tau1, tau2, e, i, g, U, tau_rec, R
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau1 = 0.5 (ms) <1e-9, 1e9>
    tau2 = 3.0 (ms) <1e-9, 1e9>
    e = 0.0 (mV)
    U = 0.0
    tau_rec = 500.0 (ms) <1e-9, 1e9>
}

ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    factor
}

STATE {
    A (uS)
    B (uS)
    R
}

INITIAL {
    LOCAL tp
    if (tau1/tau2 > 0.9999) {
        tau1 = 0.9999*tau2
    }
    A = 0.0
    B = 0.0
    R = 1.0
    tp = (tau1*tau2)/(tau2 - tau1)*log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1.0/factor
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = B - A
    i = g*(v - e)
}

DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
    R' = (1.0 - R)/tau_rec
}

NET_RECEIVE(weight (uS)) {
    if (R < 0.0) {
        R = 0.0
    }
    if (R > 1.0) {
        R = 1.0
    }
    A = A + weight*factor*R
    B = B + weight*factor*R
    R = R*(1.0 - U)
}
