import numpy as np
from scipy.signal import convolve2d


kb_reality = 1.38064852e-23 #(J/K) Joule per Kelvin

J_model = 1 #(Energy arbitrary unit)
kb_model = 1 #(Energy arb unit / Temp arb unit)

# then measured T in the model would have unit J_model/kb_model (with arb Temp unit)


def cbdc(spin_array, N):
    #create a N+2 2D array with cyclic boundary condition on the sides

    output = np.zeros((N+2, N+2))
    output[1:-1, 1:-1] = spin_array

    output[0,1:-1] = spin_array[-1,:]
    output[-1,1:-1] = spin_array[0,:]
    output[1:-1,0] = spin_array[:,-1]
    output[1:-1,-1] = spin_array[:,0]

    return output


def neighbour_product_sum(spin_array, N):

    cbdc_board = cbdc(spin_array, N)

    mask = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0]])

    convolved_mask = convolve2d(spin_array, mask)

    output = np.multiply(cbdc_board, convolved_mask).sum()

    return output


#==============================================================================================
#==============================================================================================

class spin_array():

    def __init__(self, N=3, J=J_model, kb=kb_model, muH=0, random = False):

        self.N = N
        self.J = J_model
        self.kb = kb_model
        self.muH = muH
        self.sites = self.N**2

        if random:
            zero_one_array = np.random.randint(2, size=(N,N))
        else:
            zero_one_array = np.zeros((N,N))
        
        self.array = zero_one_array*2 - 1

    def get_array(self):
        return self.array.copy()

    def flip_site(self, ij):
        i, j = ij
        self.array[i,j] *= -1
        return self

    def get_E(self):
        E = -self.J*neighbour_product_sum(self.get_array(), self.N) - self.muH*self.get_array().sum()
        return E

    def flip_delta_E(self, ij):

        i, j = ij

        current_array = self.get_array().copy()
        current_array_cbdc = cbdc(current_array, self.N)

        site_ij = current_array[i,j]

        nearest_neighbour_sum = (current_array_cbdc[i+1,j] +
                                current_array_cbdc[i+1,j+2] +
                                current_array_cbdc[i,j+1] +
                                current_array_cbdc[i+2,j+1])

        delta_E = 2*(self.J*site_ij*nearest_neighbour_sum + self.muH*site_ij)

        return delta_E

    def update_site(self, ij, T):
        
        i, j = ij

        delta_E = self.flip_delta_E(ij)

        if delta_E <= 0:
            self.flip_site(ij)
            return None
        else:
            if np.random.rand() < np.exp(-delta_E/T):
                self.flip_site(ij)
                return None
            else:
                return None

    def update_array(self, T, random=True):

        if random:
            for n in range(self.sites):

                position = np.random.randint(self.sites)

                i, j = position//self.N, position%self.N
                self.update_site((i,j), T)

        else:
            for n in range(self.sites):

                i, j = n//self.N, n%self.N
                self.update_site((i,j), T)
    
