import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.notebook import tqdm, trange
import matplotlib.animation as animation

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

def hex2np(hex_str, N):
    bi_str = bin(int(hex_str,0))[2:]
    sites = N**2

    if len(bi_str)<sites:
        bi_str = '0'*(sites-len(bi_str))+bi_str

    #array with 1,0
    array01 = np.array(list(bi_str), dtype=int)

    #array with 1,-1
    array1_1 = array01*2-1

    return array1_1.reshape(N, N)

#==============================================================================================
#==============================================================================================

class spin_array():

    def __init__(self, N=64, J=J_model, kb=kb_model, muH=0, randomseed = None, hex_input=None):

        self.N = N
        self.J = J_model
        self.kb = kb_model
        self.muH = muH
        self.sites = self.N**2

        if randomseed and hex_input != None:
            raise NameError("Array initialisation clash")

        if hex_input != None:
            self.load_hex(hex_input)

        elif randomseed != None:
            np.random.seed(randomseed)
            zero_one_array = np.random.randint(2, size=(N,N))
            self.array = zero_one_array*2 - 1
        else:
            zero_one_array = np.zeros((N,N), dtype=int)
            self.array = zero_one_array*2 - 1 

    def set_array(self, input_array):
        if input_array.shape == (self.N, self.N):
            self.array = input_array.copy()
        else:
            raise NameError('shape should be '+ str((self.N, self.N)) + ' but got ' + str(input_array.shape))

    def get_array(self):
        return self.array.copy()

    def get_E(self):
        E = -self.J*neighbour_product_sum(self.get_array()) - self.muH*self.get_array().sum()
        return E

    def min_E(self):

        min_array = np.ones((self.N,self.N))

        E = -self.J*neighbour_product_sum(min_array) - self.muH*min_array.sum()
        return E

    def get_M(self):
        return self.get_array().sum()

    def get_hex_str(self):
        #1d array with 1 and -1
        np_array_1d = self.get_array().flatten()
        #convert 1,-1 to 1,0
        np_array_1d = (np_array_1d+1)//2
        x = self.sites + 10
        bi_str = np.array2string(np_array_1d, separator='', max_line_width = x, threshold = x)[1:-1]
        hex_str = hex(int(bi_str,2))

        return hex_str

    def load_hex(self, hex_str):
        
        self.set_array(hex2np(hex_str, self.N))

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

    # an array that tells you whether to flip the site or not (True or False)
    def flip_bool_array(self, T):

        random_array = np.random.rand(self.N, self.N)
        output = random_array < np.exp(-self.flip_delta_E_array()/T)

        return output
    
    def update_array(self, T):

        #update our precomputed to the new current spin array
        flip_statement_array = self.flip_bool_array(T)
        #reset our affected list
        affected = np.zeros((self.N, self.N))

        for n in range(self.sites):

            position = np.random.randint(self.sites)

            i, j = position//self.N, position%self.N

            if affected[i,j] == 0:

                if flip_statement_array[i,j]:

                    #flipping site ij
                    self.array[i,j] *= -1

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

    def plot(self):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(self.get_array())

txt_location = 'data\\'

class hex_series():

    def __init__(self, T=1, N=64, J=J_model, kb=kb_model, muH=0, randomseed=None, initial_hex=None):

        self.N = N
        self.J = J_model
        self.kb = kb_model
        self.muH = muH
        self.sites = self.N**2
        self.T = T
        self.hex_list = []

        if randomseed != None:
            if initial_hex!=None:
                raise NameError('initial conditions settings clash')
            self.append_array(spin_array(N=N, randomseed=randomseed))
        elif initial_hex == None:
            pass
        else:
            self.append_hex(initial_hex)

    def get_para(self):
        output = 'N={}, T={}, muH={}'.format(self.N, self.T, self.muH)
        return(output)

    def append_array(self, new_spin_array):
        if not type(new_spin_array) is spin_array:
            raise NameError('This is a' + str(type(new_spin_array)) + ',not ' + str(type(spin_array)))
        
        if not new_spin_array.N == self.N:
            raise NameError('N is wrong:'+ str(new_spin_array.N))
        

        self.hex_list.append(new_spin_array.get_hex_str())

    def append_hex(self, hex_str):
        self.hex_list.append(hex_str)

    def evolve(self, frames, bar=False):

        if len(self.hex_list) ==0:
            raise NameError('its empty')
            
        if len(self.hex_list) >= frames:
            raise NameError('its already looong enough')

        last_hex = self.hex_list[-1]

        obj = spin_array(N=self.N, J=self.J, kb=self.kb, muH=self.muH, hex_input=last_hex)

        if bar:
            for i in trange(frames-len(self.hex_list)):
                obj.update_array(self.T)
                self.append_array(obj)
        else:
            for i in range(frames-len(self.hex_list)):
                obj.update_array(self.T)
                self.append_array(obj)


    def txt_file_name(self, suffix='', frames = None):
        name_T = str(int(self.T*100)).zfill(3)
        name_muH = str(self.muH).zfill(3)
        name_N = str(self.N).zfill(3)

        if frames ==None:
            name_frames = str(len(self.hex_list))
        else:
            name_frames = str(int(frames))

        file_name = txt_location+'N='+name_N+'_T='+name_T+'_muH='+name_muH+'_frames='+name_frames+suffix+'.txt'

        return file_name


    def save_txt(self, suffix=''):

        file_string = '\n'.join(self.hex_list)

        with open(self.txt_file_name(suffix=suffix), 'w') as f:
            f.write(file_string)

    def get_E_list(self):

        obj = spin_array(N=self.N, J=self.J, kb=self.kb, muH=self.muH)
        E_list = []

        for i in range(len(self.hex_list)):
            obj.load_hex(self.hex_list[i])
            E_list.append(obj.get_E())
        
        return E_list

    def get_M_list(self):

        obj = spin_array(N=self.N, J=self.J, kb=self.kb, muH=self.muH)
        M_list = []

        for i in range(len(self.hex_list)):
            obj.load_hex(self.hex_list[i])
            M_list.append(obj.get_M())
        
        return M_list



    def get_len(self):
        return len(self.hex_list)

    def get_frame_hex(self, frame):
        return self.hex_list[frame]
    
    def get_frame_np(self, frame):
        return hex2np(self.hex_list[frame], self.N)

    def min_E(self):
        obj = spin_array(N=self.N, J=self.J, kb=self.kb, muH=self.muH)
        return obj.min_E()

    def load_txt(file_name):

        para = {}
        target = {'N', 'T', 'H'}
        digits = '0123456789'

        for i in range(len(file_name)):

            if file_name[i] in target:
                
                para[file_name[i]] = file_name[i+2:i+5]

        T = int(para['T'])/100
        N = int(para['N'])
        muH = int(para['H'])
        
        series = hex_series(T=T, N=N, muH=muH)

        with open(file_name, 'r') as f:
            series.hex_list = f.readlines()
        
        return series
