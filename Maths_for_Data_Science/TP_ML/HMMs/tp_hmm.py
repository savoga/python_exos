'''
************ MARKOV CHAINS ***************************************************
'''

import numpy as np
import matplotlib.pyplot as plt

dic={'1' : ' ', '2' : 'a', '3' : 'b', '4': 'c', '5' : 'd', '6' : 'e', '7': 'f', '8' : 'g', '9' : 'h', '10': 'i', '11': 'j',
'12' : 'k', '13' : 'l', '14': 'm', '15' : 'n', '16' : 'o', '17': 'p', '18' : 'q', '19' : 'r' , '20': 's', '21' : 't', '22'
: 'u', '23': 'v', '24' : 'w', '25' : 'x' , '26': 'y', '27' : 'z', '28' : ' ' }

filename_A= 'bigramenglish.txt'
transition_matrix_en = np.loadtxt(filename_A)

'''
For each alphabetic letter, display the most frequent transition
'''

print('Most frequent transitions:')
for idx, key in enumerate(dic.keys()):
    idx_max = np.argmax(transition_matrix_en[idx])
    print("letter '{}': --> '{}' (proba = {})".format(dic[key], dic[str(idx_max+1)], np.max(transition_matrix_en[idx])))



'''
Generate state (t+1) from current state (t) using transition matrix and cdf
Display cdf
'''

def nextState(transition_matrix, dic, state):
    proba_state = transition_matrix[int(state)-1]
    next_state = np.random.choice(list(dic.keys()), p=proba_state)
    return proba_state, str(next_state)

proba_state, gen_state = nextState(transition_matrix_en, dic, '1')
print("random letter from state 1: {}".format(dic[gen_state]))

plt.title('Cumulative distribution function (from state 1)')
cumsum = np.cumsum(proba_state)
cumsum = np.array([0.0]+list(cumsum))
plt.plot(['0'] + list(dic.keys()), cumsum) # cdf
plt.savefig('cdf_state_1.jpg')

'''
Generate words using final state 28
'''

def genere_state_seq(transition_matrix, dic, final_state):
    generated_states = []
    state = '1'
    while(state != final_state):
        _, state = nextState(transition_matrix, dic, state)
        generated_states.append(state)
    return generated_states

def display_seq(dic, states):
    elements = []
    for state in states:
        elements.append(dic[state])
    return ''.join(elements)

word_1 = genere_state_seq(transition_matrix_en, dic, '28')
word_2 = genere_state_seq(transition_matrix_en, dic, '28')
word_3 = genere_state_seq(transition_matrix_en, dic, '28')
word_4 = genere_state_seq(transition_matrix_en, dic, '28')
word_5 = genere_state_seq(transition_matrix_en, dic, '28')
print(display_seq(dic, word_1))
print(display_seq(dic, word_2))
print(display_seq(dic, word_3))
print(display_seq(dic, word_4))
print(display_seq(dic, word_5))

'''
Generate sequences ending with a dot (= sentences)
We add a new state with value '.'
Probability to transit to this state from final state is 0.1
Modification of the matrix and dictionary accordingly
'''

def modifyMatAndDic(transition_matrix, dic):
    dic_2 = dic.copy()

    # modify dic
    dic_2['28'] = '' # since P(28 -> 1) = 0.9
    dic_2['29'] = '.'

    # add row
    transition_matrix_2 = np.vstack((transition_matrix, np.zeros((1,28))))

    # add column
    transition_matrix_2 = np.hstack((transition_matrix_2, np.zeros((29,1))))

    # add probabilities
    transition_matrix_2[27,0]=0.9 # probability to go from final state '' to beginning of word ' '
    transition_matrix_2[27,28]=0.1 # probability to go from end of word to dot '.'
    transition_matrix_2[27,27]=0.0
    transition_matrix_2[28,28]=1.0

    return transition_matrix_2, dic_2

transition_matrix_en_dot, dic_dot = modifyMatAndDic(transition_matrix_en, dic)

sentence_1 = genere_state_seq(transition_matrix_en_dot, dic_dot, '29')
sentence_2 = genere_state_seq(transition_matrix_en_dot, dic_dot, '29')
sentence_3 = genere_state_seq(transition_matrix_en_dot, dic_dot, '29')
sentence_4 = genere_state_seq(transition_matrix_en_dot, dic_dot, '29')
sentence_5 = genere_state_seq(transition_matrix_en_dot, dic_dot, '29')
print(display_seq(dic_dot, sentence_1))
print(display_seq(dic_dot, sentence_2))
print(display_seq(dic_dot, sentence_3))
print(display_seq(dic_dot, sentence_4))
print(display_seq(dic_dot, sentence_5))

'''
Compute likelihood to recognize the language using French/English transition
matrices of sentences "to be or not to be." and its translation
Use - for initial state of word
Use + for final state of word
Use . for final state of sentence
'''

filename_AA= 'bigramfrancais.txt'
transition_matrix_fr = np.loadtxt(filename_AA)

dic_inv = {v: k for k, v in dic_dot.items()}

def processSentence(sentence):
    EOS = False
    new_sentence = "-"
    for idx, letter in enumerate(sentence):
        if(idx == len(sentence)-1):
            EOS = True
        if(sentence[idx] == ' '):
            new_sentence += '-'
        elif(EOS and letter == '.'):
            new_sentence = new_sentence + '+' + letter
        elif(EOS):
            new_sentence = new_sentence + letter + '+'
        else:
            new_sentence += letter
        if(not EOS and sentence[idx+1] == ' '):
            new_sentence += '+'
    return new_sentence

def computeLikelihood(sentence, transition_matrix):
    new_sentence = processSentence(sentence) # transfrom sentences with + and -
    state_prev = '1'
    proba_list = []
    for letter in new_sentence[1:]:
        if letter == '-':
            state = '1'
        elif letter == '+':
            state = '28'
        else:
            state = dic_inv[letter]
        proba = transition_matrix[int(state_prev)-1, int(state)-1]
        proba_list.append(proba)
        state_prev = state
    return proba_list

transition_matrix_fr_dot, dic_dot = modifyMatAndDic(transition_matrix_fr, dic)
sentence_en = "to be or not to be"
sentence_fr = "etre ou ne pas etre"

print("***** {} *****".format(sentence_en))
proba_fr = computeLikelihood(sentence_en, transition_matrix_fr_dot)
print("likelihood (fr): {}".format(np.prod(np.array(proba_fr))))
proba_en = computeLikelihood(sentence_en, transition_matrix_en_dot)
print("likelihood (en): {}".format(np.prod(np.array(proba_en))))
# We notice that the likelihood is higher using English transition probabilities.

print("***** {} *****".format(sentence_fr))
proba_fr = computeLikelihood(sentence_fr, transition_matrix_fr_dot)
print("likelihood (fr): {}".format(np.prod(np.array(proba_fr))))
proba_en = computeLikelihood(sentence_fr, transition_matrix_en_dot)
print("likelihood (en): {}".format(np.prod(np.array(proba_en))))
# We notice that the likelihood is higher using French transition probabilities.