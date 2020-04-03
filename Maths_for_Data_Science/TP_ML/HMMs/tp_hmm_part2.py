'''
************ HMMs ************************************************************
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

filename = 'matrice_symboles.txt'
v = np.loadtxt('data_txt_compact/' + filename) # pixel configurations (columns)

filename = 'A0.txt'
A = np.loadtxt('data_txt_compact/' + filename) # transition matrix (from known state to hidden state)

filename = 'B0.txt'
B = np.loadtxt('data_txt_compact/' + filename) # observation probability matrix (from hidden state to observation)

filename = 'vect_pi0.txt'
pi = np.loadtxt('data_txt_compact/' + filename) # probabilities for initial state

'''
Explanation of zero coefficients
'''

# zero (i,j) in matrix B means that it's impossible to go from known state i (observation) to hidden state j
# zero (i,j) in matrix A means that it's impossible to go from hidden state i to hidden state j
# zero (i) in vector pi means that it's impossible to be at state i as initial state

'''
Generate state (t+1) from current state (t) using transition matrix and cdf
Display cdf
'''

def nextState(transition_matrix, state):
    proba_state = transition_matrix[state-1]
    proba_cum = np.cumsum(proba_state)
    n = np.random.random()
    next_state = len(np.where(proba_cum < n)[0])+1
    return next_state, proba_cum

initial_state = 1 # state sequence always starts from state 1
state_test = 2
q_next, cumsum = nextState(A, state_test)
fig = plt.figure()
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
cumsum = np.array([0.0]+list(cumsum))
plt.plot(np.arange(6), cumsum)
plt.title('Cumulative distribution function from state = ' + str(state_test))
plt.savefig('cdf_part_2.jpeg')

'''
Generate observation sequences for the HMM of number 0
'''

def generateStates(initial_matrix, transition_matrix):
    state_prev = np.where(np.max(initial_matrix))[0][0]+1
    state_sq = []
    for i in range(v.shape[1]):
        state_sq.append(state_prev)
        state, _ = nextState(transition_matrix, state_prev)
        state_prev = state
    return state_sq

def observationsFromStates(observation_matrix, state_sq):
    observation_sq = []
    for state in state_sq:
        proba_observation = observation_matrix[:,state-1]
        proba_cum = np.cumsum(proba_observation)
        n = np.random.random()
        observation = len(np.where(proba_cum < n)[0])+1
        observation_sq.append(observation)
    return observation_sq

'''
Display images for 0, 1 and 7
'''

number_list = ['0','1','7']
fig, ax = plt.subplots(figsize=(6,6))
k = 1
for number in number_list:
    B = np.loadtxt('data_txt_compact/' + 'B' + number + '.txt')
    A = np.loadtxt('data_txt_compact/' + 'A' + number + '.txt')
    pi = np.loadtxt('data_txt_compact/' + 'vect_pi' + number + '.txt')
    for i in range(5):
        state_sq = generateStates(pi, A)
        observation_sq = observationsFromStates(B, state_sq)
        im=[]
        for t in range(0,len(observation_sq)):
            im_col = v[:, observation_sq[t]-1]
            im.append(im_col)

        im = np.array(im).T # now make an array
        plt.subplot(3,5,k)
        if((k+3) % 5 == 0):
            plt.title('generation of number {}'.format(number))
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
        plt.imshow(im*255, cmap='Greys', interpolation='none', aspect='auto')
        k += 1
plt.savefig("number_generation.jpg")

'''
Compute likelihood using Viterbi algorithm
'''

def likelihoodViterbi(seq, A, B, pi):

    A = np.ma.log(A).filled(-500)
    B = np.ma.log(B).filled(-500)
    pi = np.ma.log(pi).filled(-500)
    # Note: log is used because if we have long sequences, the final coefficient
    # would be too close to zero (since it's a product) and there is a risk that
    # we loose precision. Sing the log allows us to work with sum so the final
    # coefficient would be higher.
    # We can use exponential to find the true value afterward

    vit_matrix = np.zeros((A.shape[0], len(seq)))
    # initialization
    for state in range(1,vit_matrix.shape[0]+1):
        observation = seq[0]
        proba_start = pi[state-1]
        proba_obs = B[state-1,observation-1]
        vit_matrix[state-1,0] = proba_start + proba_obs
    # fill the rest of the matrix
    state_list = np.arange(vit_matrix.shape[0])+1
    for obs_idx in range(len(seq[1:])):
        observation = seq[obs_idx]
        for state in state_list:
            proba_trajectories = []
            for state_prev in state_list:
                delta_prev = vit_matrix[state_prev-1,obs_idx]
                proba_obs = B[observation-1,state-1]
                proba_state = A[state_prev-1,state-1]
                proba_trajectories.append(delta_prev + proba_obs + proba_state)
            vit_matrix[state-1,obs_idx+1] = np.max(proba_trajectories)
    return np.max(vit_matrix[:,-1])

likelihood_list = []
filename = 'SeqTest7.txt'
seq = np.loadtxt('data_txt_compact/' + filename)
seq_int = seq.astype(int)
for seq in seq_int:
    likelihood_number = []
    for number in number_list:
        A = np.loadtxt('data_txt_compact/' + 'A' + number + '.txt')
        B = np.loadtxt('data_txt_compact/' + 'B' + number + '.txt')
        pi = np.loadtxt('data_txt_compact/' + 'vect_pi' + number + '.txt')
        likelihood_number.append(likelihoodViterbi(seq, A, B, pi))
    likelihood_list.append(likelihood_number) # likelihood list of size (10, 3)

'''
Classify the test sequences using Viterbi
'''

for idx_seq, likelihood_model in enumerate(likelihood_list):
    idx_model = np.argmax(likelihood_model)
    model = None
    if(idx_model == 0):
        model = 0
    if(idx_model == 1):
        model  = 1
    if(idx_model == 2):
        model = 7
    print("Sequence {}: model prediction {}".format(idx_seq+1, model))
