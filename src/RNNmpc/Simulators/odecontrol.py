import numpy as np
import torch
import scipy.integrate


class OdeControl:
    """Simulation object for ODE systems.

    Attributes:
    -----------
    control_disc: float
        control discretization
    rtol: float
        relative tolerance for scipy.integrate.solve_ivp
    atol: float
        absolute tolerance for scipy.integrate.solve_ivp
    default_x0: np.array
        reasonable initial condition from which to simulate

    Methods:
    -----------
    control_step(U,x0)
        advance system one control timestep
    simulate(U, x0)
        simulate system under control input U
    generate_data(num_trajectories,Tfinal)
        generate training data for network
    """

    def __init__(self, dynamics: function, x0: np.array, nu: int, control_disc: float = 0.01) -> None:
        """Initialize system with specified control discretization.

        No model discretization is used because we use the variable time-stepper in
        scipy.integrate.solve_ivp.

        Parameters:
        -----------
        dynamics: function
            dynamics function to be used. 
        x0: np.array
            initial condition
        nu: int
            size of control vector
        control_disc: float
            control discretization
        """
        self.dynamics = dynamics
        self.control_disc = control_disc
        self.rtol = 1e-12
        self.atol = 1e-12
        self.default_x0 = torch.tensor(x0, dtype=torch.float64)
        self.nu = nu
        self.nx = len(x0)

        try:
            if np.shape(dynamics(0, x0, np.zeros(nu))) != np.shape(x0):
                raise ValueError("initial conditions vector is not the correct length")
        except:
            raise ValueError("initial conditions vector is not the correct length")
        
        try: 
            dynamics(0,x0,np.zeros(nu))
            try: 
                dynamics(0,x0,np.zeros(nu+1))
            except:
                None
        except:
            raise ValueError("nu is too small")

    def control_step(self, U: np.array, x0: np.array) -> np.array:
        """Advance system one control step.

        Parameters:
        ----------
        U: np.array
            control for control model step, dims (Nu, 1)
        x0: np.array
            initial condition for control step, dims (3,1)

        Returns:
        ----------
        x0: np.array
            system state advanced by one control step, dims (3,1)
        """
        x0 = x0.flatten()
        sol = scipy.integrate.solve_ivp(
            self.dynamics,
            t_span=[0, self.control_disc + 1e-8],
            y0=x0,
            args=(U,),
            rtol=self.rtol,
            atol=self.atol,
            t_eval=np.arange(0, 2 * self.control_disc, self.control_disc),
        )

        return sol.y[:, -1]

    def simulate(
        self, U: torch.DoubleTensor, x0: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """Simulate the system under given control U.

        Parameters:
        -----------
        U: torch.DoubleTensor
            control inputs for simulation, dims (Nu, t_steps)
        x0: torch.DoubleTensor
            initial condition, dims (3, 1)

        Returns:
        ----------
        X_list: torch.DoubleTensor
            evolution of system under control inputs, dims (3, t_steps)
        """
        U = U.detach().numpy()
        x0 = x0.detach().numpy()
        t_steps = U.shape[1]
        X_list = np.empty((x0.size, t_steps))
        for step in range(t_steps):
            x0 = self.control_step(U[:, step], x0)
            X_list[:, step] = x0
        X_list = torch.tensor(X_list, dtype=torch.float64)
        return X_list

    def generate_data(self, num_trajectories, Tfinal):
        """Generate random trajectories for training a neural network

        Parameters:
        -----------
        num_trajectories: int
            number of trajectories to generate
        Tfinal: torch.DoubleTensor
            final time for the trajectories

        Returns:
        ----------
        trajectories: torch.DoubleTensor
            n trajectories, dims (num_trajectories, len(x0), len(Tvec))
        """
        n = num_trajectories
        Tvec = np.arange(0,Tfinal,self.control_disc)
        default_x0 = self.default_x0
        trajectories = torch.zeros((n,default_x0.size,np.size(Tvec)))


        #generate n random x0,u
        for i in range(0,n):
            print(default_x0.size())
            x0 = torch.rand(default_x0.shape)
            u = torch.rand((self.nu,len(Tvec)))
            trajectories[i,:] = self.simulate(u,x0)
            #self.simulate(u,x0)
        #simulate n each x0,u
        #return trajectories
        #how should it be returned?
        return trajectories
    
    def set_default_x0(self, x0: np.array):
        try:
            if np.shape(self.dynamics(0, x0, np.zeros(self.nu))) != np.shape(x0):
                raise ValueError("initial conditions vector is not the correct length")
        except:
            raise ValueError("initial conditions vector is not the correct length")
        self.default_x0 = x0