import numpy as np
from scipy.signal import convolve2d


kb_reality = 1.38064852e-23 #(J/K) Joule per Kelvin

J_model = 1 #(Energy arbitrary unit)
kb_model = 1 #(Energy arb unit / Temp arb unit)

# then measured T in the model would have unit J_model/kb_model (with arb Temp unit)


def cbdc(spin_array):
    #create a N+2 2D array with cyclic boundary condition on the sides

    height, width = spin_array.shape

    output = np.zeros((height+2, width+2))
    output[1:-1, 1:-1] = spin_array

    output[0,1:-1] = spin_array[-1,:]
    output[-1,1:-1] = spin_array[0,:]
    output[1:-1,0] = spin_array[:,-1]
    output[1:-1,-1] = spin_array[:,0]

    return output


def neighbour_product_sum(spin_array):

    cbdc_board = cbdc(spin_array)

    mask = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0]])

    convolved_mask = convolve2d(spin_array, mask)

    output = np.multiply(cbdc_board, convolved_mask).sum()

    return output


#==============================================================================================
#==============================================================================================

class spin_array():

    def __init__(self, N=32, J=J_model, kb=kb_model, muH=0, randomseed = None, array_input=None):

        self.N = N
        self.J = J_model
        self.kb = kb_model
        self.muH = muH
        self.sites = self.N**2

        if randomseed and array_input != None:
            raise NameError("Array initialisation clash")

        if array_input != None:
            if array_input.shape == (N,N):
                self.array = array_input
            else:
                raise NameError('input array has wrong dimensions')

        elif randomseed != None:
            np.random.seed(randomseed)
            zero_one_array = np.random.randint(2, size=(N,N))
            self.array = zero_one_array*2 - 1
        else:
            zero_one_array = np.zeros((N,N))
            self.array = zero_one_array*2 - 1 


    #return two dimensional indices i and j from a 1D index (0~N-1)
    def n2ij(self, n):
        i, j = n//self.N, n%self.N

        return (i,j)

    def ij2n(self,ij):
        i, j = ij
        n = self.N*i + j

        return n

    def get_array(self):
        return self.array.copy()

    def flip_site(self, ij):
        i, j = ij
        self.array[i,j] *= -1
        return self

    def get_E(self):
        E = -self.J*neighbour_product_sum(self.get_array()) - self.muH*self.get_array().sum()
        return E

    def get_M(self):
        return self.get_array().sum()

    def flip_delta_E(self, ij):

        i, j = ij

        current_array = self.get_array().copy()
        current_array_cbdc = cbdc(current_array)

        site_ij = current_array[i,j]

        nearest_neighbour_sum = (current_array_cbdc[i+1,j] +
                                current_array_cbdc[i+1,j+2] +
                                current_array_cbdc[i,j+1] +
                                current_array_cbdc[i+2,j+1])

        delta_E = 2*(self.J*site_ij*nearest_neighbour_sum + self.muH*site_ij)

        return delta_E


    def flip_delta_E_array(self):

        spin_array = self.get_array()

        cbdc_board = cbdc(spin_array)

        mask = np.array([[0,1,0],
                        [1,0,1],
                        [0,1,0]])

        convolved_mask = convolve2d(spin_array, mask)

        product = np.multiply(cbdc_board, convolved_mask)

        reduced = product[1:-1, 1:-1]

        reduced[0,:] += product[-1, 1:-1]
        reduced[-1,:] += product[0, 1:-1]
        reduced[:,0] += product[1:-1,-1]
        reduced[:,-1] += product[1:-1,0]

        delta_E_array = 2*(self.J*reduced + self.muH*spin_array)
        

        return delta_E_array


    def update_site(self, ij, T):
        
        i, j = ij

        delta_E = self.flip_delta_E(ij)

        if np.random.rand() < np.exp(-delta_E/T):
            self.flip_site(ij)

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

    # an array that tells you whether to flip the site or not (True or False)
    def flip_bool_array(self, T):

        random_array = np.random.rand(self.N, self.N)
        output = random_array < np.exp(-self.flip_delta_E_array()/T)

        return output
    
    #update the whole array by flipping the sites according to the metropolis algorithm (speed up)
    def update_array_beta(self, T):

        flip_statement_array = self.flip_bool_array(T)

        # an array which store a 1 for all the flipping and neighbouring sites
        affected = np.zeros((self.N, self.N))

        chosen_array = np.zeros((self.N, self.N), dtype=int)

        for n in range(self.sites):

            i, j = np.random.randint(self.N), np.random.randint(self.N)

            # if the site chosen is an affected site
            if affected[i,j]:

                final_flipping = (flip_statement_array*chosen_array)+(np.ones((self.N, self.N))-chosen_array)

                self.array = self.array*(flip_statement_array*chosen_array)+(np.ones((self.N, self.N))-chosen_array)    
                

                #update our precomputed to the new current spin array
                flip_statement_array = self.flip_bool_array(T)

                #reset
                chosen_array = np.zeros((self.N, self.N))
                affected = np.zeros((self.N, self.N))


            chosen_array[i,j] = 1

            #infects the neighbouring sites and itself
            affected[i,j] = 1
            affected[i-1,j] = 1
            affected[(i+1)-self.N,j] = 1
            affected[i,j-1] = 1
            affected[i,(j+1)-self.N] = 1
        
        self.array = self.array*(flip_statement_array*chosen_array)+(np.ones((self.N, self.N))-chosen_array)    


    def update_array_new(self, T):

        #update our precomputed to the new current spin array
        flip_statement_array = self.flip_bool_array(T)
        #reset our affected list
        affected = np.zeros((self.N, self.N))

        for n in range(self.sites):

            position = np.random.randint(self.sites)

            i, j = position//self.N, position%self.N

            if affected[i,j] == 0:

                if flip_statement_array[i,j]:

                    self.flip_site((i,j))

                    affected[i,j] = 1
                    affected[i-1,j] = 1
                    affected[(i+1)-self.N,j] = 1
                    affected[i,j-1] = 1
                    affected[i,(j+1)-self.N] = 1

            #if the position that we chose is next to or is a site that we flipped
            else:

                #update our precomputed to the new current spin array
                flip_statement_array = self.flip_bool_array(T)
                #reset our affected list
                affected = np.zeros((self.N, self.N))



        

def sma(input_list, n):
    sma_mask = np.ones(n)/n
    output = np.convolve(input_list, sma_mask)[:1-n]
    return output

def eq_time(E_list, bins=50):
    histogram, bin_edges = np.histogram(E_list, bins=bins)
    peak_value = bin_edges[np.argmax(histogram)+1]

    eq_steps = None

    for i in range(len(E_list)):
        if E_list[i]<peak_value:
            eq_steps = i
            break
    
    if eq_steps == None:
        raise NameError('eq_steps not found somehow')

    return eq_steps