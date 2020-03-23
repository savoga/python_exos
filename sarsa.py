# action is identified with new state (after move)

def sarsa(Q, model = model, alpha = 0.1, eps = 0.1, n_iter = 100):
    terminals = model.terminals
    rewards = model.rewards
    gamma = model.gamma
    # random state (not terminal)
    state = np.random.choice(np.setdiff1d(np.arange(len(model.states)), terminals))
    # random action
    action = np.random.choice(Q[state].indices)
    new_state = action
    for t in range(n_iter):
        state_prev = state
        action_prev = action
        state = new_state
        if state in terminals:
            # restart
            state = np.random.choice(np.setdiff1d(np.arange(len(model.states)), terminals)) # on veut pas d'état terminal ?
            action = np.random.choice(Q[state].indices)
            Q[state_prev, action_prev] = (1 - alpha) *  Q[state_prev, action_prev] + alpha * rewards[action_prev]
        else:
            best_action = Q[state].indices[np.argmax(Q[state].data)]
            if np.random.random() < eps:
                action = np.random.choice(Q[state].indices)
            else:
                action = best_action
            Q[state_prev, action_prev] = (1 - alpha) *  Q[state_prev, action_prev] + alpha * (rewards[action_prev] + gamma * Q[state, action])
        new_state = action
    return Q
