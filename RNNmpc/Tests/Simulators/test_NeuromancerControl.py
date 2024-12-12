from neuromancer import psl
import numpy as np
#from NeuromancerControl import neuromancerControl
from RNNmpc.Simulators.NeuromancerControl import neuromancerControl


#write smokes tests


def test_generate_data():
    """
    Test function for the generate_data method in the
    NeuromancerControl class

    Attributes:None
    Inputs:None
    Outputs:None
    """
    #simple spring mass damper system to test
    Sim = neuromancerControl(psl.nonautonomous.Actuator())

    train_len = 5.0
    switching_period = 0.1
    filter_len = 5
    dist_min = -1.0
    dist_max = 1.0
    train_percentage = 70.0

    data = Sim.generate_data(train_len = train_len,
                             switching_period = switching_period,
                             filter_len = filter_len, #maybe this should be 0?
                             dist_min = dist_min,
                             dist_max = dist_max,
                             train_percentage = train_percentage)
    
    U_train = data['U_train']
    U_valid = data['U_valid']
    S_train = data['S_train']
    S_valid = data['S_valid']
    O_train = data['O_train']
    O_valid = data['O_valid']

    #test return dictionary has the right datatypes and values
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

    datalen = round(train_len/Sim.model_disc)
    utrainlen = round(datalen*train_percentage/100)
    xtrainlen = utrainlen
    uvalidlen = round(datalen*(100-train_percentage)/100)
    xvalidlen = uvalidlen-1
    nu = Sim.nu
    nx = Sim.nx

    #check sizes
    print(np.shape(U_train))
    print(utrainlen)
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

