import numpy as np
import torch
import scipy as sp
from RNNmpc.Simulators.odeControl import odeControl



def test_generate_data():
    """
    Test of the generate_data function of odeControl
    Inputs:None
    Outputs:None
    """
    #simple spring mass damper system to test
    def dynamics(t,x,u):
        xdot = np.zeros(np.size(x))
        xdot[0] = x[1]
        xdot[1] = -2*x[1]-2*x[0]+u[0]

        return xdot

    x0 = np.array([1,1])
    sim = odeControl(dynamics,x0,1)
    
    train_len = 5.0
    switching_period = 0.1
    filter_len = 5
    dist_min = -1.0
    dist_max = 1.0
    train_percentage = 70.0
    x0_max = np.array([1,1])
    x0_min = np.array([-1,-1])

    data = sim.generate_data(train_len = train_len,
                             switching_period = switching_period,
                             filter_len = filter_len, #maybe this should be 0?
                             dist_min = dist_min,
                             dist_max = dist_max,
                             x0_min = x0_min,
                             x0_max = x0_max,
                             train_percentage = train_percentage)
    

    U_train = data['U_train']
    U_valid = data['U_valid']
    S_train = data['S_train']
    S_valid = data['S_valid']
    O_train = data['O_train']
    O_valid = data['O_valid']


    #test if output is the right dimensions and datatype

    assert train_len == data['train_len']
    assert switching_period == data['switching_period']
    assert filter_len == data['filter_len']

    #check datatypes
    assert isinstance(U_train,type([]))
    assert isinstance(U_valid,type([]))
    assert isinstance(S_train,type([]))
    assert isinstance(S_valid,type([]))
    assert isinstance(O_train,type([]))
    assert isinstance(O_valid,type([]))

    datalen = round(train_len/sim.control_disc)
    print(sim.control_disc)
    utrainlen = round(datalen*train_percentage/100)
    xtrainlen = utrainlen
    uvalidlen = round(datalen*(100-train_percentage)/100)
    xvalidlen = uvalidlen-1
    nu = sim.nu
    nx = sim.nx

    #check sizes
    assert np.shape(U_train) == (nu,utrainlen)
    assert np.shape(U_valid) == (nu,uvalidlen)
    assert np.shape(S_train) == (nx,xtrainlen)
    assert np.shape(S_valid) == (nx,xvalidlen)
    assert np.shape(O_train) == (nx,xtrainlen)
    assert np.shape(O_valid) == (nx,xvalidlen)

    #check that S and O are timeshifted
    S_train = np.asarray(S_train)
    S_valid = np.asarray(S_valid)
    O_train = np.asarray(O_train)
    O_valid = np.asarray(O_valid)
    assert np.array_equal(S_train[:,1:], O_train[:,:-1])
    assert np.array_equal(S_valid[:,1:], O_valid[:,:-1])

def test_oneshot():
    """
    Testing against a known solution for initial condition x0
    and step input u

    Inputs:None
    Outputs:None
    
    """
    #simple spring mass damper system to test
    def dynamics(t,x,u):
        xdot = np.zeros(np.size(x))
        xdot[0] = x[1]
        xdot[1] = -2*x[1]-2*x[0]+u

        return xdot
    
    def analytical_sol(t,x0):
        A = np.array([[0,1],[-2,-2]])
        ynat = x0
        for k in t[1:]:
            sol = sp.linalg.expm(A*k)@x0
            ynat = np.vstack((ynat,sol))
        y1 = np.e**(-t)*(-(1/2)*np.sin(t)-(1/2)*np.cos(t))+1/2
        y2 = np.e**(-t)*(1*np.sin(t)+0*np.cos(t))
        y = np.vstack((y1,y2))+ynat.T
        return y

    T = 10
    x0 = np.array([1,1])

    sim = odeControl(dynamics,x0,1,control_disc=0.01)
    t = np.arange(0,T,sim.control_disc)
    u = np.ones(len(t))
    u = u.reshape(1,len(t))

    ysim = sim.simulate(torch.DoubleTensor(u),torch.DoubleTensor(x0))
    ysol = analytical_sol(t,x0)

    error = np.abs(ysim-ysol)
    error = error.numpy()

    assert np.any(error < 1e-8)