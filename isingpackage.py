import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from copy import deepcopy #create deep copies of an object
from tqdm.notebook import tqdm, trange #progress bars to track the progress of the codes
from numba import jit #just in time compilation, to speed up the heavy computation parts

J_model = 1 #(Energy arbitrary unit)
kb_model = 1 #(Energy arb unit / Temp arb unit)

# Measurements of T in the model would have unit (J_model/kb_model)

# Measurements of the strength of the external magnetic field H would be in units of J/mu
# where mu is the magnetic moment of each site. We will express most of our H strength with muH
# which is the product between H and mu, in the unit of J_model

def cbdc(spin_array):
    """cbdc: creates a N+2*N+2 2D array with cyclic boundary condition on the all sides


    Args:
        spin_array (2d numpy array)

    Returns:
        2d numpy array
    """    
    height, width = spin_array.shape

    output = np.zeros((height+2, width+2))
    output[1:-1, 1:-1] = spin_array

    output[0,1:-1] = spin_array[-1,:]
    output[-1,1:-1] = spin_array[0,:]
    output[1:-1,0] = spin_array[:,-1]
    output[1:-1,-1] = spin_array[:,0]

    return output


def neighbour_product_sum(spin_array):
    """neighbour_product_sum: calculates and returns SUM(spin_i, spin_j) for all neighbouring ij pairs in the array (under cyclic boundary condition)

    Args:
        spin_array (2D np array)

    Returns:
        float
    """    

    cbdc_board = cbdc(spin_array)

    mask = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0]])

    convolved_mask = convolve2d(spin_array, mask)

    output = np.multiply(cbdc_board, convolved_mask).sum()

    return output

def hex2np(hex_str, N):
    """hex2np converts a hexadecimal number into a square numpy array with 1 and -1, corresponding to 1 and 0 in the binary value of the hexadecimal (see spin_array.get_hex_str for the detail of the conversion)

    Args:
        hex_str (string): a hexadecimal number with the prefix 0x

    Returns:
        2D np array
    """    
    #removing the 0x prefix and converting it into binary
    bi_str = bin(int(hex_str,0))[2:]
    sites = N**2

    #if the leading digits of the binary are 0s, we put the 0s back
    if len(bi_str)<sites:
        bi_str = '0'*(sites-len(bi_str))+bi_str

    #array with 1,0
    array0_1 = np.array(list(bi_str), dtype=int)

    #array with 1,-1
    array1_1 = array0_1*2-1

    return array1_1.reshape(N, N)

#==============================================================================================

@jit
def update_array_jit(input_array, T, muH=0):
    """update_array_jit updates the 2D spin array by carrying out N*N monte-carlo steps according to the metropolis algorithm with the given temperature and H-field, on randomly selected sites. For jit compilation to work, the function is required to have use no global variables.

    Args:
        input_array (np array): the 2D spin array
        T (float): Temperature of the surrounding
        muH (float, optional): The H-field strength(given in muH since it's strength depends on the ratio of mu/J). Defaults to 0.

    Returns:
        np array: an array after carrying out the monte-carlo steps
    """    

    N = input_array.shape[0]

    #N*N monte carlo steps
    for n in range(N**2):

        #select a random site
        position = np.random.randint(N**2)
        i, j = position//N, position%N

        #the spin of the chosen site (+1 or -1)
        site_ij = input_array[i,j]

        #sum of the spins of the four neighbours of the chosen site
        nearest_neighbour_sum = (input_array[(i+1)%N,j] +
                                input_array[(i-1)%N,j] +
                                input_array[i,(j+1)%N] +
                                input_array[i,(j-1)%N])

        #the change in total energy if the chosen site is flipped
        delta_E = 2*(site_ij*nearest_neighbour_sum + muH*site_ij)

        #deciding whether the site would be flipped according to the metropolis algorithm
        if np.random.rand() < np.exp(-delta_E/T):
            input_array[i,j] *= -1
    
    #the final spin array
    return input_array


#==============================================================================================

#A class to store our square 2d spin-array with +1 and -1
class spin_array():

    def __init__(self, N=64, J=J_model, kb=kb_model, randomseed = None, hex_input=None):

        self.N = N #width of the square lattice
        self.J = J_model #the coupling strength of the neighbouring sites (default to 1)
        self.kb = kb_model #boltzman constant of the model (default to 1)
        self.sites = self.N**2 #number of sites in the array

        #given two initial conditions
        if randomseed != None and hex_input != None:
            raise NameError("Array initialisation clash")

        #loads a hex string as its spin array configuration
        if hex_input != None:
            self.load_hex(hex_input)

        #generates a random 2d array as its configuration
        elif randomseed != None:
            np.random.seed(randomseed)
            zero_one_array = np.random.randint(2, size=(N,N))
            self.array = zero_one_array*2 - 1
        # just a 2d array with all zeros as its configuration
        else:
            zero_one_array = np.zeros((N,N), dtype=int)
            self.array = zero_one_array*2 - 1 

    def set_array(self, input_array):
        """set_array sets the configuration to the input np array

        Args:
            input_array (2d np array)

        Raises:
            NameError: if input_array has the wrong shape
        """        
        if input_array.shape == (self.N, self.N):
            self.array = input_array.copy()
        else:
            raise NameError('shape should be '+ str((self.N, self.N)) + ' but got ' + str(input_array.shape))

    def get_array(self):
        """get_array returns the array configuration

        Returns:
            np array
        """        
        return self.array.copy()

    def get_E(self, muH=0):
        """get_E returns the total energy of the system.

        Args:
            muH (float, optional): The strength of the H-field. Defaults to 0.

        Returns:
            _type_: _description_
        """        
        E = -self.J*neighbour_product_sum(self.get_array()) - muH*self.get_array().sum()
        return E

    def min_E(self, muH=0):
        """min_E returns the lowest possible total energy of the system (when all the spins are aligned)

        Args:
            muH (int, optional): H field strength. Defaults to 0.

        Returns:
            float: lowest possible total energy
        """        

        min_array = np.ones((self.N,self.N))

        E = -self.J*neighbour_product_sum(min_array) - abs(muH*self.sites)
        return E

    def get_M(self):
        """get_M returns the total magnetic moment of the system 

        Returns:
            float: the measurement is in unit of mu
        """        
        return self.get_array().sum()
    
    def get_M_mean(self):
        """get_M_mean returns the mean magnetic moment per site of the system 

        Returns:
            float: the measurement is in unit of mu
        """        
        return self.get_array().sum()/self.sites

    def get_hex_str(self):
        """get_hex_str converts the configuration of the array into a hexadecimal number by converting the +1, -1 array into a 1, 0 bitmap, and express the binary number representation of the bitmap as a hexadecimal.

        Returns:
            string: a hexadecimal that starts with the prefix 0x
        """        
        #1d array with 1 and -1
        np_array_1d = self.get_array().flatten()
        #convert 1,-1 to 1,0
        np_array_1d = (np_array_1d+1)//2
        x = self.sites + 10
        bi_str = np.array2string(np_array_1d, separator='', max_line_width = x, threshold = x)[1:-1]
        hex_str = hex(int(bi_str,2))

        return hex_str

    def load_hex(self, hex_str):
        """load_hex setting the configuration of the spin array according to a hex string.

        Args:
            hex_str (string): a hexadecimal string with prefix 0x
        """
        self.set_array(hex2np(hex_str, self.N))

    def update_array(self, T, muH=0):
        """update_array updates the entire array through the metropolis temperature, which depends on the external temperature and magnetic field. For details, please look at the update_array_jit function's docstring.

        Args:
            T (float): External temperature
            muH (float, optional): external magnetic field. Defaults to 0.
        """        
        #The most computationally heavy part of the simulation
        #therefore, using a precompiled function to speed it up.
        self.array = update_array_jit(self.get_array(), T, muH=muH)
        

    def plot(self):
        """plot function plots the configuration of the spin array by calling the matpotlib.pyplot.imshow function
        """        
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(self.get_array())


#the location to store all the generated data
txt_location = 'data\\'

#a series class to store every frame of the evolution of a system as hex strings
class hex_series():

    def __init__(self, T=1, N=64, J=J_model, kb=kb_model, muH=0, randomseed=None, initial_hex=None):

        #the same parameters as the class spin_array
        self.N = N
        self.J = J_model
        self.kb = kb_model
        self.sites = self.N**2

        #the evolution of the system is held under constant T and H, therfore we can have them as variables of the series.
        self.T = T
        self.muH = muH

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
        """get_para returns the N, T, and muH of the system in the series
        """        
        output = 'N={}, T={}, muH={}'.format(self.N, self.T, self.muH)
        return(output)

    def append_array(self, new_spin_array):
        """append_array adds a new frame to the series

        Args:
            new_spin_array (class spin_array): spin_array object holding the configuration of the array that we want to append.

        Raises:
            NameError: if the type of the input is wrong
            NameError: if the spin_array that we're trying to append is not the right shape
        """        
        if not type(new_spin_array) is spin_array:
            raise NameError('This is a' + str(type(new_spin_array)) + ',not ' + str(type(spin_array)))
        
        if not new_spin_array.N == self.N:
            raise NameError('N is wrong:'+ str(new_spin_array.N))
        
        #adds the configuration of the array as a hex string
        self.hex_list.append(new_spin_array.get_hex_str())

    def append_hex(self, hex_str):
        """append_array adds a new frame to the series

        Args:
            hex_str (string): a string of a hexadecimal holding the configuration with prefix 0x
        """          
        self.hex_list.append(hex_str)

    def evolve(self, frames, bar=False):
        """evolve evolves the configuration of the system from the last frame in the series using the spin_array.update_array() method.

        Args:
            frames (int): the target length of the series
            bar (bool, optional): Whether we want progress bar or not. Defaults to False.

        Raises:
            NameError: if the series is empty and doesn't have a last frame
            NameError: if the series is already longer or equal to the target length
        """        

        if len(self.hex_list) ==0:
            raise NameError('its empty')
            
        if len(self.hex_list) >= frames:
            raise NameError('its already looong enough')

        last_hex = self.hex_list[-1]

        obj = spin_array(N=self.N, J=self.J, kb=self.kb, hex_input=last_hex)

        if bar:
            for i in trange(frames-len(self.hex_list)):
                obj.update_array(self.T)
                self.append_array(obj)
        else:
            for i in range(frames-len(self.hex_list)):
                obj.update_array(self.T)
                self.append_array(obj)

    def txt_file_name(self, suffix='', frames = None):
        """txt_file_name returns a string of a filename according to a systematic naming rule. eg. 

        Args:
            suffix (str, optional): a suffix option if we want have multiple data with the same parameters. Defaults to ''.
            frames (int, optional): If the function is being used to save the current series to a txt file, the function will automatically fill in this number according to the length of the series. If we're using this function to get the name of the file with the parameters we want, we would have to fill in the frames variable ourselves. Defaults to None.

        Returns:
            string: the file name
        """        
        name_T = str(round(self.T*100)).zfill(3)
        name_muH = str(self.muH).zfill(3)
        name_N = str(self.N).zfill(3)

        #number of frames not given
        if frames ==None:
            name_frames = str(len(self.hex_list))

        else:
            name_frames = str(int(frames))

        file_name = txt_location+'N='+name_N+'_T='+name_T+'_muH='+name_muH+'_frames='+name_frames+suffix+'.txt'

        return file_name


    def save_txt(self, suffix=''):
        """save_txt saves the file as a txt to our desinated location


        Args:
            suffix (str, optional): Defaults to ''.
        """        

        #seperating our list of hex strings with a new line character '\n'
        file_string = '\n'.join(self.hex_list)

        with open(self.txt_file_name(suffix=suffix), 'w') as f:
            f.write(file_string)

    def get_E_list(self):
        """get_E_list returns the list of total energy of each frame

        Returns:
            list of float: a list of total energy of each frame
        """
        #a dummy object of spin_array to call the method get_E()
        obj = spin_array(N=self.N, J=self.J, kb=self.kb)
        E_list = []

        for i in range(len(self.hex_list)):
            obj.load_hex(self.hex_list[i])
            E_list.append(obj.get_E(muH=muH))
        
        return E_list

    def get_M_list(self):
        """get_M_list returns the list of total magnetic moment of each frame

        Returns:
            list of float: a list of total magnetic moment of each frame
        """

        obj = spin_array(N=self.N, J=self.J, kb=self.kb)
        M_list = []

        for i in range(len(self.hex_list)):
            obj.load_hex(self.hex_list[i])
            M_list.append(obj.get_M())
        
        return M_list



    def get_len(self):
        """get_len returns the length of the series. (The number of frames it holds)

        Returns:
            int: length of self.hex_list
        """        
        return len(self.hex_list)

    def get_frame_hex(self, frame):
        """get_frame_hex returns the configuration of a specific frame as a string of hex

        Args:
            frame (int): the index of the frame that we want to get

        Returns:
            string: the configuration of the array at that frame in hex string.
        """        
        return self.hex_list[frame]
    
    def get_frame_np(self, frame):
        """get_frame_np returns the configuration of a specific frame as a np array

        Args:
            frame (int): the index of the frame that we want to get

        Returns:
            string: the configuration of the array at that frame in np array
        """        

        return hex2np(self.hex_list[frame], self.N)

    def min_E(self):
        """min_E returns the minimum energy of the system. See spin_array.min_E for details

        Returns:
            float
        """        
        obj = spin_array(N=self.N, J=self.J, kb=self.kb)
        return obj.min_E(self.muH)

    def load_txt(file_name):
        """load_txt creates a hex_series object by reading a text file that is saved by the .save_txt() method

        Args:
            file_name (str): the file name of the file that we want to read

        Returns:
            hex_series
        """
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
