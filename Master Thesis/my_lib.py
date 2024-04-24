'''
Library of functions used in the QA notebooks for my Master's Thesis.

Inspired by the work of Sergi Ramos-Calderer, Carlos Bravo-Prieto, 
Ruge Lin, Emanuele Bellini, Marc Manzano, Najwa Aaraj, and José I. Latorre 
for  
    "Solving systems of Boolean multivariate equations with quantum annealing" 
(https://arxiv.org/abs/2111.13224).
'''

#################################### Imports

import numpy as np
from sympy import symbols, Subs
from pyqubo import Binary, Array

import dimod, minorminer
from dimod import BQM
from dwave.system import EmbeddingComposite, DWaveSampler
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.inspector import show

import examples_nnf as nnf

#################################### Hamiltonian construction

def building_H_direct_pyqubo(bits: int, chosen_model: str):

    '''
        Build the Hamiltonian in direct embedding, either 
        in QUBO model or BQM model, using PyQubo.

        Inputs:
            bits (int): number of variables/classical bits
            chosen_model (str): variable that determines the model to build. 
                                The possible values are QUBO or BQM.

        Outputs:
            H (bqm/qubo pyqubo obj): Hamiltonian in the chosen model
            sol (list(list)): knowns solution(s) of the given MQ system
    '''

    ## setup
    #x_bin = Array([Binary(f'x{i}') for i in range(bits)])     # variables 0-8
    x_bin = Array([Binary(f'x{i}') for i in range(1,bits+1)])  # variables 1-9
    #display(x_bin)

    if bits == 5:
        p_bin, sol = nnf.example_5(*x_bin)      # system def + known solution
    elif bits == 9:
        p_bin, sol = nnf.example_9(*x_bin)      # system def + known solution
    print(f'Actual solution(s) for the MQ problem:\n{sol}')

    print('\n-----------------------------------------------------\n')

    ## Hamiltonian
    H_bin = sum(p_bin)
    model_H = H_bin.compile()
    if chosen_model == 'QUBO':
        print('You chose the QUBO model.')
        H = model_H.to_qubo()
    elif chosen_model == 'BQM':
        print('You chose the BQM model.')
        H = model_H.to_bqm() 
    else:
        print('You can only choose QUBO or BQM')

    print('\n-----------------------------------------------------')

    return(H, sol)


#################################### D-Wave Running


def single_it_from_pyqubo(H, topology, chosen_chainstrength):

    '''
        Function that runs *once* the sampling for the given Hamiltonian
        on D-Wave devices.

        Inputs:
            H (BQM model): problem Hamiltonian from pyqubo building.
            topology (str): string variable identifying the architecture on
                            which will run the problem. Either "Zephyr" or "Pegasus".
            chosen_chainstrength (str): choesen chain_stregth for the run. 
                                  Either "Article" (definition from the starting article) 
                                  or "Ocean_Doc" (definition from Ocean Documentation).

        Outputs:
            response: SampleSet object with the details of the run. 
    '''

    ## parameters (same as the article)
    numruns = 1000
    T = 20

    ## sampler
    if topology == "Zephyr":
        sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage2_prototype2.2'}))
        print('Running on Zephyr Topology.')
    elif topology == "Pegasus":
        sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage_system6.3'}))
        print('Running on Pegasus Topology.')

    ## chainstrength
    if chosen_chainstrength == 'Ocean_Doc':
        chain_strength = uniform_torque_compensation(H, sampler)
        print(f'You chose the chainstrength from the Ocean Documentation: {chain_strength}')
    elif chosen_chainstrength == 'Article':
        chain_strength = 0
        H_qubo = H.to_qubo()
        for i in H_qubo[0].values():
            chain_strength += abs(i)
        chain_strength *= 3/len(H_qubo[0])
        print(f'You chose the Article chainstrength: {chain_strength}.')
    else:
        print('You can ony choose Ocean_Doc or Article.')

    ## sampling
    if type(H) != dimod.binary.binary_quadratic_model.BinaryQuadraticModel:
        print('You must use a BQM model.')
    else:
        response = sampler.sample(H, chain_strength=chain_strength, 
                                num_reads=numruns, annealing_time=T, 
                                answer_mode='histogram',
                                label = 'mq_on_zephyr')
    print('Finished running the experiment!')
        
    return(response)

class dwave_runners:
    ''' 
        Class that implements the iterative procedure 
        and the single run of a D-Wave annealing.
    '''

    def __init__(self, Q, offset, bits, topology, chosen_chainstrength, numruns = 1000, T=20):
        '''
            Initialize the variables for the class.

            Inputs:
                Q (dict): QUBO model.
                offset (float): offset of the QUBO model.
                bits (int): number of variables.
                topology (str): chosen topology, either "Zephyr" or "Pegasus".
                chosen_chainstrength (str): chosen definition, either "Article" or "Ocean_Doc".
                numruns (int): number of sampling per run (default = 1000).
                T (int): annealing time (default = 20).
        '''

        self.bits = bits
        self.numruns = numruns
        self.T = T
        self.topology = topology
        self.chosen_chainstrength = chosen_chainstrength
        self.energies_hist = []
        self.Q = Q
        self.offset = offset

        self.into_BQM()
        self.define_topology()


    def into_BQM(self):
        '''
            Build the BQM model from the QUBO model in input.
        '''
        print('Building the BQM model.')
        self.H = BQM.from_qubo(self.Q, self.offset)


    def define_topology(self):
        '''
            Define the D-Wave sampler, with automated embedding, 
            specifying the wanted topology.
        '''

        if self.topology == "Zephyr":
            self.sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage2_prototype2.2'}))
            print('Running on Zephyr Topology.')
        elif self.topology == "Pegasus":
            self.sampler = EmbeddingComposite(DWaveSampler(solver={'name': 'Advantage_system6.3'}))
            print('Running on Pegasus Topology.')
        

    def define_chainstrength(self):
        '''
            Calculate the chosen_chainstrength following the definition from the original article
            or using the definition suggested from the Ocean documentation.
        '''

        if self.chosen_chainstrength == 'Ocean_Doc':
            self.chain_strength = uniform_torque_compensation(self.H, self.sampler)
            print(f'You chose the chainstrength from the Ocean Documentation: {self.chain_strength}')
        elif self.chosen_chainstrength == 'Article':
            self.chain_strength = 0
            for i in self.Q.values():
                self.chain_strength += abs(i)
            self.chain_strength *= 3/len(self.Q)
            print(f'You chose the Article chainstrength: {self.chain_strength}.')
        else:
            print('You can only choose Ocean_Doc or Article.')
    

    def decoding_response(self, response):
        '''
            Decode the SampleSet obtained from a D-Wave run and
            turn it into a list of values.

            Inputs:
                response: SampleSet object containing the data

            Outputs:
                solution (list(int)): best sample in list format
        '''

        best_sample = response.first.sample
        solution = []

        for i in range(self.bits):
            solution.append(best_sample[i])
        print(f"Solution found: {solution} with energy: {response.first.energy}")

        return(solution)
    

    def get_info_on_embedding(self, used_embedding):

        '''
            Get info on embedding of a single run on a QPU.

            Inputs:
                used_embedding (dict): mapping of logical qubits onto physical ones.

            Outputs:
                physical_qubits (int): number of physical qubits used in the embedding.
                sorted_chains (dict): logical qubits paired with their chainlength. Sorted by key.
        '''
        q_count = 0
        chains = {}

        for logical_q in used_embedding.keys():
            q_count += len(used_embedding[logical_q])
            chains[f'lq_{logical_q}'] = len(used_embedding[logical_q])
        
        sorted_chains = dict(sorted(chains.items()))

        return(q_count, sorted_chains)
    

    def single_it(self):
        '''
            Function that runs *once* the sampling for the given Hamiltonian
            on D-Wave devices.

            Outputs:
                response: SampleSet object with the details of the sampling. 
        '''

        self.define_chainstrength()      
        response = self.sampler.sample(self.H, chain_strength=self.chain_strength, 
                              num_reads=self.numruns, annealing_time=self.T, 
                              answer_mode='histogram',
                              label = f'mq_on_{self.topology}')
        print('Finished running the experiment!')

        return(response)


    def iterative(self, iterations=5, threshold=10):

        '''
            Run on QPU multiple times, each iteration fixing 
            ancillae that share the same value.

            Inputs: 
                iterations (int): number of single runs (default=5).
                threshold (int): number of samples to comparate to fix the ancillae (default=10).

            Outputs:
                solution (list): list of final found values for the variables.
                timing_info (dict): dictionary with total timings.
                physical_qubits (float): total number of the physical qubits used.
                it (int): final iteration run.
        '''

        ## general setup     
        fixed = {}          # variable : value

        ###### iterating
        for it in range(iterations):

            ## info per run
            num_variables = self.H.num_variables
            print(f'Number of variables: {num_variables}')
                    
            ## running on QPU
            response = self.single_it()
            record = response.record
            order = np.argsort(record['energy'])        ## needed 'cause the samples are not ordered

            best_sample, best_energy = response.first.sample, response.first.energy
            #best_sample, best_energy = record.sample[order[0]], record.energy[order[0]]
            print(f'Energy of best sample at iteration {it}: {best_energy}')
            #print(f'Best sample: {best_sample}')

            ## info
            if it == 0:                 # first iteration
                #timings
                timing_info = response.info['timing']
                #embedding
                embedding_info = response.info['embedding_context']['embedding']
                physical_qubits, chains = self.get_info_on_embedding(embedding_info)
                show(response)
            else:                        # later iterations
                # timings
                for key in timing_info.keys():
                    timing_info[key] += response.info['timing'][key]
                # embedding
                ph_info = response.info['embedding_context']['embedding']
                ph_qubits, ph_chains = self.get_info_on_embedding(ph_info)
                physical_qubits += ph_qubits

            ## energies histogram
            ph_energy = []
            for i in range(len(record.energy)):
                for j in range(record.num_occurrences[order[i]]):
                    ph_energy.append(record.energy[order[i]])

            self.energies_hist.append(ph_energy)

            if best_energy == 0:
                print(f'Solution found with final iteration {it}.')
                #show(response)
                break
            
            ## fixing variables
            print("Fixing ancillae...")
            for i in range(self.bits, num_variables):
                flag = True
                ph_value = record.sample[order[0]][i]

                # checking for shared values
                for k in range(min(threshold, len(record.sample))):
                    if ph_value != record.sample[order[k]][i]:
                        flag = False
                        break
                if flag:
                    fixed[i] = ph_value
            
            print(f'Fixed {len(fixed.keys())} qubits:')
            for key in fixed.keys():
                print(f'{key} : {fixed[key]}')
            print('\n------------------------------------------------------------------------\n')

            ## updating Hamiltonian
            for var in fixed.keys():
                if var not in self.H.variables:        # checking if it's a new fix or not
                    continue
                else:
                    self.H.fix_variable(var, fixed[var])

            ## creating new QUBO model
            self.Q, self.offset = self.H.to_qubo()

        ###### reconstruncting
        print('Reconstructing the final state...')
        solution = []
        for i in range(self.bits):
            solution.append(best_sample[i])

        print(f'Solution found: {solution}')

        return(solution, timing_info, physical_qubits, it)

    def counting_run(self):

        '''
            Single run for counting qubits. It's defined to not calculate multiple 
            times the chainstrength and to avoid useless prints.

            Outputs:
                response = SampleSet object with the details of the run.
        '''
    
        response = self.sampler.sample(self.H, chain_strength=self.chain_strength, 
                              num_reads=self.numruns, annealing_time=self.T, 
                              answer_mode='histogram',
                              label = f'mq_on_{self.topology}')

        return(response)


    def counting_qubits(self, average):

        '''
            Run the sampling 'average' times to get a mean of 
            the physical qubits needed to embed the problem.

            Inputs:
                average (int): number of runs to calculate the mean on.
            
            Outputs:
                logical_qubits (float): number of used logical qubits.
                physical_qubits (float): mean of used physical qubits.
        '''

        logical_qubits = 0
        physical_qubits = 0

        self.numruns = 1
        self.define_chainstrength()      

        for t in range(average):

            response = self.counting_run()
            print(f'Finished running the experiment #{t+1}!')

            used_embedding = response.info['embedding_context']['embedding']
            ph_qubits, chains = self.get_info_on_embedding(used_embedding=used_embedding)

            physical_qubits += ph_qubits
            logical_qubits += len(used_embedding.keys())

        print('\n--------------------------------------------------------------------------\n')

        return(logical_qubits/average, physical_qubits/average)
