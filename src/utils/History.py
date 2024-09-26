################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Implementation of History class <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################






class History:
    """ It is used to collect info during training.
    """

    def __init__(self):
        self.container = dict()



    def store(self, key, value):
        """ Appends "value" to the list of the container specified by "key".

        Args:
            key: the dictionary key to extract the needed list.
            value: the value to append.
        
        """
        if not(key in set(self.container.keys())):
            self.container[key] = list()
        self.container[key].append(value)



    def init_last(self, in_dict, id, coeff : float):
        """ Stores the elements of "in_dict" in the History container.

        Args:   
            in_dict: the input dictionary.
            id: the key identifier.
            coeff (float): a multiplicative constant. 

        """
        for key in in_dict.keys():
            self.store(id + "_" + key, in_dict[key] * coeff)



    def update_last(self, in_dict, id, coeff : float):
        """ Updates the last stored elements with "in_dict" inputs.

        Args:   
            in_dict: the input dictionary.
            id: the key identifier.
            coeff (float): a multiplicative constant. 

        """
        for key in in_dict.keys():
            self.container[id + "_" + key][-1] += (in_dict[key] * coeff)
    


    def get_last_values(self):
        """ Gets all the last values collected in the container 

        Returns:
            The last values collected in the container 
        """
        output_dict = {
            key : self.container[key][-1] for key in self.container.keys() 
        }
        return output_dict
    

    
    def __bool__(self):
        return self.container.__bool__
