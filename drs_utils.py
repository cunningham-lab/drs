import numpy as np
import tensorflow as tf


def mu_rnns(x, units, length, N, d):
    l = []
    for i in xrange(N):
        with tf.variable_scope("mu" + str(i)):
            _, state = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(units), tf.expand_dims(x[i], 2), dtype=tf.float32,
                                         sequence_length=length[i])
            #             _, state = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units), tf.contrib.rnn.LSTMCell(units), tf.expand_dims(x[i], 2), dtype=tf.float32, sequence_length=length[i])
            l.append(state[1])
            #             l.append(tf.concat([state[0][1], state[1][1]], axis=1))
    l = tf.concat(l, 1)
    with tf.variable_scope("mu"):
        out = tf.layers.dense(inputs=l, units=d, activation=tf.identity)
    return out


def sigma_rnns(x, units, length, N, d):
    l = []
    for i in xrange(N):
        with tf.variable_scope("sigma" + str(i)):
            _, state = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(units), tf.expand_dims(x[i], 2), dtype=tf.float32,
                                         sequence_length=length[i])
            #             _, state = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.LSTMCell(units), tf.contrib.rnn.LSTMCell(units), tf.expand_dims(x[i], 2), dtype=tf.float32, sequence_length=length[i])
            l.append(state[1])
            #             l.append(tf.concat([state[0][1], state[1][1]], axis=1))
    l = tf.concat(l, 1)
    with tf.variable_scope("sigma"):
        out = tf.layers.dense(inputs=l, units=d, activation=tf.identity)
        out = tf.exp(out)
    return out


def reshape_for_params(x, s1, s2, N, L, n_intervals):
    # reshapes the output of the f neural network to correspond with our spline parametrization
    # [N, None, L, n_intervals, s, s]
    Q1s = x[:, :((s1 ** 2) * n_intervals * N)]
    Q1s = tf.transpose(tf.reshape(Q1s, [-1, L, N, n_intervals, s1, s1]), [2, 0, 1, 3, 4, 5])
    Q1s = Q1s + tf.transpose(Q1s, [0, 1, 2, 3, 5, 4])  # make sure the matrices are symmetric
    Q2s = x[:, (N * (s1 ** 2) * n_intervals):]
    Q2s = tf.transpose(tf.reshape(Q2s, [-1, L, N, n_intervals, s2, s2]), [2, 0, 1, 3, 4, 5])
    Q2s = Q2s + tf.transpose(Q2s, [0, 1, 2, 3, 5, 4])  # make sure the matrices are symmetric
    return Q1s, Q2s


def spline_spd_proj(As, s):
    # As has shape [a0, a1, a2, a3, s, s]
    # projects each s x s symmetric matrix in As to the SPD cone
    if s > 1:
        Ds, Vs = tf.self_adjoint_eig(As)
        Ps = tf.matmul(Vs, tf.matmul(tf.matrix_diag(tf.nn.relu(Ds)), tf.transpose(Vs, [0, 1, 2, 3, 5, 4])))
    else:
        Ps = tf.nn.relu(As)
    return Ps


def tridiagonal_solve(As_l, As_d, As_u, rhs, size):
    # As_l, As_d and As_u have respective shapes [a0, a1, a2, size-1], [a0, a1, a2, size] and [a0, a1, a2, size-1]
    # rhs has shape [a0, a1, a2, size]
    # As_l, As_d and As_u correspond, respectively, to the lower diagonal, diagonal and upper diagonal parts
    # of tridiagonal matrices As of shape [a0, a1, a2, size, size]
    # output has shape [a0, a1, a2, size+2], output[:,:,:,1:-1] is the solution to As * x = rhs (batching over a0, a1 and a2)
    # and output[:,:,:,0] = output[:,:,:,-1] = 0
    sol = []
    aux1 = []
    aux2 = []
    for i in xrange(size):
        if i == 0:
            aux1.append(As_u[:, :, :, i] / As_d[:, :, :, i])
            aux2.append(rhs[:, :, :, i] / As_d[:, :, :, i])
        else:
            if i < size - 1:
                denominator = As_d[:, :, :, i] - As_l[:, :, :, i - 1] * aux1[-1]
                aux1.append(As_u[:, :, :, i] / denominator)
                aux2.append((rhs[:, :, :, i] - As_l[:, :, :, i - 1] * aux2[-1]) / denominator)
            else:
                denominator = As_d[:, :, :, i] - As_l[:, :, :, i - 1] * aux1[-1]
                aux2.append((rhs[:, :, :, i] - As_l[:, :, :, i - 1] * aux2[-1]) / denominator)
    aux1 = tf.transpose(aux1, [1, 2, 3, 0])
    aux2 = tf.transpose(aux2, [1, 2, 3, 0])
    for i in xrange(size + 2):
        if i == 0 or i == size + 1:
            sol.append(tf.zeros_like(As_d[:, :, :, 0]))
        else:
            if i == 1:
                sol.append(aux2[:, :, :, size - i])
            else:
                sol.append(aux2[:, :, :, size - i] - aux1[:, :, :, size - i] * sol[-1])
    sol = tf.reverse(tf.transpose(sol, [1, 2, 3, 0]), axis=[3])
    return sol


def lambda_continuity(Q1s, Q2s, p, s2, N, L, n_intervals, brackets, bracket_mat, cs):
    # returns the Lagrange multipliers from the KKT conditions associated with the spline continuity projection problem
    # make sure the first and last values of lambda are 0 (along n_intervals+1), shape is [N,None,L,n_intervals+1]
    repeated_bracket = tf.tile(
        tf.reshape(brackets[1:-1], [1, 1, 1, np.shape(brackets[1:-1])[0], np.shape(brackets[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    if p % 2 == 0:
        As_l = 0.5 * tf.matmul(tf.matmul(tf.expand_dims(brackets[2:-1], 1), bracket_mat[1:-2, :, :]),
                               tf.expand_dims(brackets[2:-1], 2))
        As_d = -tf.matmul(tf.matmul(tf.expand_dims(brackets[1:-1], 1), bracket_mat[1:-1, :, :]),
                          tf.expand_dims(brackets[1:-1], 2))
        As_u = 0.5 * tf.matmul(tf.matmul(tf.expand_dims(brackets[1:-2], 1), bracket_mat[2:-1, :, :]),
                               tf.expand_dims(brackets[1:-2], 2))
        rhs = tf.matmul(
            tf.matmul(tf.expand_dims(repeated_bracket, 4), Q1s[:, :, :, 1:, :, :] - Q1s[:, :, :, :-1, :, :]),
            tf.expand_dims(repeated_bracket, 5))
        rhs = tf.reshape(rhs, [N, -1, L, tf.shape(rhs)[3]])
        As_l = tf.tile(tf.reshape(As_l, [1, 1, 1, tf.shape(As_l)[0]]), [N, tf.shape(Q1s)[1], L, 1])
        As_d = tf.tile(tf.reshape(As_d, [1, 1, 1, tf.shape(As_d)[0]]), [N, tf.shape(Q1s)[1], L, 1])
        As_u = tf.tile(tf.reshape(As_u, [1, 1, 1, tf.shape(As_u)[0]]), [N, tf.shape(Q1s)[1], L, 1])
        lam = tridiagonal_solve(As_l, As_d, As_u, rhs, n_intervals - 1)
    else:
        expanded_cs = tf.tile(tf.reshape(cs, [1, 1, 1, np.shape(cs)[0], 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s2, s2])
        expanded_cs_small = tf.tile(tf.reshape(cs, [1, 1, 1, np.shape(cs)[0]]), [N, tf.shape(Q1s)[1], L, 1])
        numerator = tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket, 4),
                                        expanded_cs * Q2s[:, :, :, :-1, :, :] - Q1s[:, :, :, 1:, :, :]),
                              tf.expand_dims(repeated_bracket, 5))
        numerator = tf.reshape(numerator, [N, -1, L, n_intervals - 1])
        denominator = tf.matmul(tf.expand_dims(brackets[1:-1], 1), tf.expand_dims(brackets[1:-1], 2)) ** 2
        denominator = tf.tile(tf.reshape(denominator, [1, 1, 1, n_intervals - 1]), [N, tf.shape(Q1s)[1], L, 1])
        denominator = (1 + expanded_cs_small ** 2) / 2 * denominator
        sol = numerator / denominator
        lam = []
        # add a first and last zero coordinates to lambda:
        for i in xrange(n_intervals + 1):
            if i == 0 or i == n_intervals:
                lam.append(tf.zeros_like(sol[:, :, :, 0]))
            else:
                lam.append(sol[:, :, :, i - 1])
        lam = tf.transpose(lam, [1, 2, 3, 0])
    return lam


def bracket_x(x, d):
    # x has shape [n]
    # gives back output of shape [n, d], out[i,:] = [x[i]^0, x[i]^1, ..., x[i]^(d-1)]
    # currently implemented in numpy for fixed nodes, has to be implemented in tensorflow for nodes to be learnt
    b = np.power(np.expand_dims(x, 1), np.arange(d), dtype='float32')
    return b


def spline_continuity_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals):
    # projects the piece-wise polynomials parametrized by Q1s and Q2s onto the space of continuous piece-wise polynomials
    # bracket_mat has shape [n_intervals+1, s1, s1], bracket_mat[i,:,:] corresponds to bracket(node[i])*bracket(node[i])'
    cs = (nodes[1:-1] - nodes[:-2]) / (nodes[2:] - nodes[1:-1])
    expanded_cs = tf.tile(tf.reshape(cs, [1, 1, 1, np.shape(cs)[0], 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s2, s2])
    brackets = bracket_x(nodes, s1)
    bracket_mat = tf.matmul(tf.expand_dims(brackets, 2), tf.expand_dims(brackets, 1))
    lam = lambda_continuity(Q1s, Q2s, p, s2, N, L, n_intervals, brackets, bracket_mat, cs)
    repeated_bracket_mat = tf.tile(tf.reshape(bracket_mat, [1, 1, 1, tf.shape(bracket_mat)[0], tf.shape(bracket_mat)[1],
                                                            tf.shape(bracket_mat)[2]]),
                                   [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    expanded_lam = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s1, s1])
    if p % 2 == 0:
        Q1s_new = Q1s + expanded_lam[:, :, :, :-1] / 2 * repeated_bracket_mat[:, :, :, :-1, :, :] - expanded_lam[:, :,
                                                                                                    :,
                                                                                                    1:] / 2 * repeated_bracket_mat[
                                                                                                              :, :, :,
                                                                                                              1:, :, :]
        Q2s_new = Q2s
    else:
        # add a 0 to expanded_cs
        new_cs = []
        for i in xrange(n_intervals):
            if i == n_intervals - 1:
                new_cs.append(tf.zeros_like(expanded_cs[:, :, :, 0]))
            else:
                new_cs.append(expanded_cs[:, :, :, i])
        new_cs = tf.transpose(new_cs, [1, 2, 3, 0, 4, 5])
        Q1s_new = Q1s + expanded_lam[:, :, :, :-1] / 2 * repeated_bracket_mat[:, :, :, :-1, :, :]
        Q2s_new = Q2s - expanded_lam[:, :, :, 1:] / 2 * new_cs * repeated_bracket_mat[:, :, :, 1:, :, :]
    return Q1s_new, Q2s_new


def lambda_differentiability(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals, brackets, bracket_mat, cross_bracket_mat,
                             brackets2, bracket_mat2, brackets_p):
    # returns the Lagrange multipliers from the KKT conditions associated with the spline differentiability projection problem
    # make sure the first and last values of lambda are 0 (along n_intervals+1), shape is [N,None,L,n_intervals+1]
    repeated_bracket = tf.tile(
        tf.reshape(brackets[1:-1], [1, 1, 1, np.shape(brackets[1:-1])[0], np.shape(brackets[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket_p = tf.tile(
        tf.reshape(brackets_p[1:-1], [1, 1, 1, np.shape(brackets_p[1:-1])[0], np.shape(brackets_p[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket2 = tf.tile(
        tf.reshape(brackets2[1:-1], [1, 1, 1, np.shape(brackets2[1:-1])[0], np.shape(brackets2[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_nodes_s1 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s1, s1])
    repeated_nodes_s2 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q2s)[1], L, 1, s2, s2])
    expanded_nodes = tf.reshape(nodes, [-1, 1, 1])
    if p % 2 == 0:
        As_l1 = tf.matmul(tf.matmul(tf.expand_dims(brackets_p[2:-1], 1), cross_bracket_mat[1:-2, :, :]),
                          tf.expand_dims(brackets[2:-1], 2))
        As_l2 = - (expanded_nodes[2:-1] - expanded_nodes[1:-2]) ** 2 / 2 * tf.matmul(
            tf.matmul(tf.expand_dims(brackets2[2:-1], 1), bracket_mat2[1:-2, :, :]), tf.expand_dims(brackets2[2:-1], 2))
        As_l = As_l1 + As_l2
        As_d1 = - 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-1], 1), cross_bracket_mat[1:-1, :, :]),
                                tf.expand_dims(brackets[1:-1], 2))
        As_d2 = (- (expanded_nodes[1:-1] - expanded_nodes[:-2]) ** 2 / 2 - (
        expanded_nodes[2:] - expanded_nodes[1:-1]) ** 2 / 2) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets2[1:-1], 1), bracket_mat2[1:-1, :, :]), tf.expand_dims(brackets2[1:-1], 2))
        As_d = As_d1 + As_d2
        As_u1 = tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-2], 1), cross_bracket_mat[2:-1, :, :]),
                          tf.expand_dims(brackets[1:-2], 2))
        As_u2 = - (expanded_nodes[2:-1] - expanded_nodes[1:-2]) ** 2 / 2 * tf.matmul(
            tf.matmul(tf.expand_dims(brackets2[1:-2], 1), bracket_mat2[2:-1, :, :]), tf.expand_dims(brackets2[1:-2], 2))
        As_u = As_u1 + As_u2
        rhs1 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(repeated_bracket_p, 4), Q1s[:, :, :, 1:, :, :] - Q1s[:, :, :, :-1, :, :]),
            tf.expand_dims(repeated_bracket, 5))
        rhs2 = tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket2, 4),
                                   (repeated_nodes_s2[:, :, :, 2:] - repeated_nodes_s2[:, :, :, 1:-1]) * Q2s[:, :, :,
                                                                                                         1:, :, :] - (
                                   repeated_nodes_s2[:, :, :, :-2] - repeated_nodes_s2[:, :, :, 1:-1]) * Q2s[:, :, :,
                                                                                                         :-1, :, :]),
                         tf.expand_dims(repeated_bracket2, 5))
        rhs = rhs1 + rhs2
    else:
        As_l1 = tf.matmul(tf.matmul(tf.expand_dims(brackets[2:-1], 1), bracket_mat[1:-2, :, :] - (
        expanded_nodes[2:-1] - expanded_nodes[1:-2]) / 2 * cross_bracket_mat[1:-2, :, :]),
                          tf.expand_dims(brackets[2:-1], 2))
        As_l2 = (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[2:-1], 1), bracket_mat[1:-2, :, :]), tf.expand_dims(brackets[2:-1], 2))
        As_l = As_l1 + As_l2
        As_d1 = tf.matmul(tf.matmul(tf.expand_dims(brackets[1:-1], 1), - 2 * bracket_mat[1:-1, :, :] + (
        expanded_nodes[2:] - 2 * expanded_nodes[1:-1] + expanded_nodes[:-2]) / 2 * cross_bracket_mat[1:-1, :, :]),
                          tf.expand_dims(brackets[1:-1], 2))
        As_d2 = ((expanded_nodes[2:] - 2 * expanded_nodes[1:-1] + expanded_nodes[:-2]) - (
        expanded_nodes[1:-1] - expanded_nodes[:-2]) ** 2 - (
                 expanded_nodes[2:] - expanded_nodes[1:-1]) ** 2) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[1:-1], 1), cross_bracket_mat[1:-1, :, :]),
            tf.expand_dims(brackets[1:-1], 2))
        As_d = As_d1 + As_d2
        As_u1 = tf.matmul(tf.matmul(tf.expand_dims(brackets[1:-2], 1), bracket_mat[2:-1, :, :] + (
        expanded_nodes[2:-1] - expanded_nodes[1:-2]) / 2 * cross_bracket_mat[2:-1, :, :]),
                          tf.expand_dims(brackets[1:-2], 2))
        As_u2 = - (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[1:-2], 1), bracket_mat[2:-1, :, :]), tf.expand_dims(brackets[1:-2], 2))
        As_u = As_u1 + As_u2
        rhs1 = tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket, 4),
                                   Q1s[:, :, :, :-1, :, :] - Q1s[:, :, :, 1:, :, :] + Q2s[:, :, :, 1:, :, :] - Q2s[:, :,
                                                                                                               :, :-1,
                                                                                                               :, :]),
                         tf.expand_dims(repeated_bracket, 5))
        rhs2 = 2 * tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket_p, 4),
                                       (repeated_nodes_s1[:, :, :, 1:-1] - repeated_nodes_s1[:, :, :, :-2]) * Q1s[:, :,
                                                                                                              :, 1:, :,
                                                                                                              :] - (
                                       repeated_nodes_s2[:, :, :, 2:] - repeated_nodes_s2[:, :, :, 1:-1]) * Q2s[:, :, :,
                                                                                                            :-1, :, :]),
                             tf.expand_dims(repeated_bracket, 5))
        rhs = rhs1 + rhs2
    rhs = tf.reshape(rhs, [N, -1, L, tf.shape(rhs)[3]])
    As_l = tf.tile(tf.reshape(As_l, [1, 1, 1, tf.shape(As_l)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    As_d = tf.tile(tf.reshape(As_d, [1, 1, 1, tf.shape(As_d)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    As_u = tf.tile(tf.reshape(As_u, [1, 1, 1, tf.shape(As_u)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    lam = tridiagonal_solve(As_l, As_d, As_u, rhs, n_intervals - 1)
    return lam


def bracket_prime_x(x, d):
    # x has shape [n]
    # gives back output of shape [n, d], out[i,:] = [0, 1 * x[i]^0, 2 * x[i]^1, ..., (d-1) * x[i]^(d-2)]
    # currently implemented in numpy for fixed nodes, has to be implemented in tensorflow for nodes to be learnt
    b = np.power(np.expand_dims(x, 1), np.concatenate((np.array([0]), np.arange(d - 1)))) * np.arange(d)
    return np.array(b, dtype='float32')


def spline_differentiability_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals):
    # projects the piece-wise polynomials parametrized by Q1s and Q2s onto the space of differentiable piece-wise polynomials
    # bracket_mat has shape [n_intervals+1, s1, s1], bracket_mat[i,:,:] corresponds to bracket(node[i])*bracket(node[i])'
    # bracket_mat2 is the analogous with bracket2, shape is now [n_intervals+1, s2, s2]
    # cross_bracket_mat has shape [n_intervals+1, s1, s1], cross_bracket_mat[i,:,:] corresponds to bracket_p(node[i])*bracket(node[i])' + bracket(node[i])*bracket_p(node[i])'
    brackets = bracket_x(nodes, s1)
    brackets2 = bracket_x(nodes, s2)
    bracket_mat = tf.matmul(tf.expand_dims(brackets, 2), tf.expand_dims(brackets, 1))
    bracket_mat2 = tf.matmul(tf.expand_dims(brackets2, 2), tf.expand_dims(brackets2, 1))
    brackets_p = bracket_prime_x(nodes, s1)
    cross_bracket_mat = tf.matmul(tf.expand_dims(brackets, 2), tf.expand_dims(brackets_p, 1)) + tf.matmul(
        tf.expand_dims(brackets_p, 2), tf.expand_dims(brackets, 1))
    lam = lambda_differentiability(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals, brackets, bracket_mat,
                                   cross_bracket_mat, brackets2, bracket_mat2, brackets_p)
    repeated_bracket_mat = tf.tile(tf.reshape(bracket_mat, [1, 1, 1, tf.shape(bracket_mat)[0], tf.shape(bracket_mat)[1],
                                                            tf.shape(bracket_mat)[2]]),
                                   [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    repeated_bracket_mat2 = tf.tile(tf.reshape(bracket_mat2,
                                               [1, 1, 1, tf.shape(bracket_mat2)[0], tf.shape(bracket_mat2)[1],
                                                tf.shape(bracket_mat2)[2]]), [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    repeated_cross_bracket_mat = tf.tile(tf.reshape(cross_bracket_mat, [1, 1, 1, tf.shape(cross_bracket_mat)[0],
                                                                        tf.shape(cross_bracket_mat)[1],
                                                                        tf.shape(cross_bracket_mat)[2]]),
                                         [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    repeated_nodes_s2 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s2, s2])
    repeated_nodes_s1 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s1, s1])
    expanded_lam_s1 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s1, s1])
    expanded_lam_s2 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s2, s2])
    if p % 2 == 0:
        Q1s_new = Q1s - expanded_lam_s1[:, :, :, 1:] / 2 * repeated_cross_bracket_mat[:, :, :, 1:, :,
                                                           :] + expanded_lam_s1[:, :, :,
                                                                :-1] / 2 * repeated_cross_bracket_mat[:, :, :, :-1, :,
                                                                           :]
        Q2s_new = Q2s - (repeated_nodes_s2[:, :, :, :-1] - repeated_nodes_s2[:, :, :, 1:]) / 2 * (
        expanded_lam_s2[:, :, :, 1:] * repeated_bracket_mat2[:, :, :, 1:, :, :] + expanded_lam_s2[:, :, :,
                                                                                  :-1] * repeated_bracket_mat2[:, :, :,
                                                                                         :-1, :, :])
    else:
        Q1s_new = Q1s + expanded_lam_s1[:, :, :, 1:] / 2 * repeated_bracket_mat[:, :, :, 1:, :, :] - expanded_lam_s1[:,
                                                                                                     :, :, :-1] / 2 * (
                                                                                                     repeated_bracket_mat[
                                                                                                     :, :, :, :-1, :,
                                                                                                     :] - (
                                                                                                     repeated_nodes_s1[
                                                                                                     :, :, :,
                                                                                                     1:] - repeated_nodes_s1[
                                                                                                           :, :, :,
                                                                                                           :-1]) * repeated_cross_bracket_mat[
                                                                                                                   :, :,
                                                                                                                   :,
                                                                                                                   :-1,
                                                                                                                   :,
                                                                                                                   :])
        Q2s_new = Q2s - expanded_lam_s2[:, :, :, 1:] / 2 * (repeated_bracket_mat[:, :, :, 1:, :, :] + (
        repeated_nodes_s2[:, :, :, 1:] - repeated_nodes_s2[:, :, :, :-1]) * repeated_cross_bracket_mat[:, :, :, 1:, :,
                                                                            :]) + expanded_lam_s2[:, :, :,
                                                                                  :-1] / 2 * repeated_bracket_mat[:, :,
                                                                                             :, :-1, :, :]
    return Q1s_new, Q2s_new


def lambda_2nd_differentiability(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals, brackets, brackets2, brackets_p,
                                 brackets_p2, brackets_pp, cross_bracket_mat, cross_2nd_bracket_mat, Bs, Cs):
    # returns the Lagrange multipliers from the KKT conditions associated with the spline 2nd differentiability projection problem
    # make sure the first and last values of lambda are 0 (along n_intervals+1), shape is [N,None,L,n_intervals+1]
    repeated_bracket = tf.tile(
        tf.reshape(brackets[1:-1], [1, 1, 1, np.shape(brackets[1:-1])[0], np.shape(brackets[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket_p = tf.tile(
        tf.reshape(brackets_p[1:-1], [1, 1, 1, np.shape(brackets_p[1:-1])[0], np.shape(brackets_p[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket_pp = tf.tile(
        tf.reshape(brackets_pp[1:-1], [1, 1, 1, np.shape(brackets_pp[1:-1])[0], np.shape(brackets_pp[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket2 = tf.tile(
        tf.reshape(brackets2[1:-1], [1, 1, 1, np.shape(brackets2[1:-1])[0], np.shape(brackets2[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_bracket_p2 = tf.tile(
        tf.reshape(brackets_p2[1:-1], [1, 1, 1, np.shape(brackets_p2[1:-1])[0], np.shape(brackets_p2[1:-1])[1]]),
        [N, tf.shape(Q1s)[1], L, 1, 1])
    repeated_nodes_s1 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q1s)[1], L, 1, s1, s1])
    repeated_nodes_s2 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1, 1]), [N, tf.shape(Q2s)[1], L, 1, s2, s2])
    repeated_nodes_s1_small = tf.tile(tf.reshape(nodes, [-1, 1, 1]), [1, s1, s1])
    repeated_nodes_s2_small = tf.tile(tf.reshape(nodes, [-1, 1, 1]), [1, s2, s2])
    expanded_nodes = tf.reshape(nodes, [-1, 1, 1])
    if p % 2 == 0:
        As_l1 = tf.matmul(tf.matmul(tf.expand_dims(brackets_pp[2:-1], 1), cross_2nd_bracket_mat[1:-2, :, :]),
                          tf.expand_dims(brackets[2:-1], 2))
        As_l2 = tf.matmul(tf.matmul(tf.expand_dims(brackets_p[2:-1], 1), cross_2nd_bracket_mat[1:-2, :, :]),
                          tf.expand_dims(brackets_p[2:-1], 2))
        As_l3 = - 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets2[2:-1], 1), Cs[1:-1, :, :]),
                                tf.expand_dims(brackets2[2:-1], 2))
        As_l4 = 4 * (expanded_nodes[1:-2] - expanded_nodes[2:-1]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p2[2:-1], 1), Cs[1:-1, :, :]), tf.expand_dims(brackets2[2:-1], 2))
        As_l = As_l1 + As_l2 + As_l3 + As_l4
        As_d1 = - 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets_pp[1:-1], 1), cross_2nd_bracket_mat[1:-1, :, :]),
                                tf.expand_dims(brackets[1:-1], 2))
        As_d2 = - 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-1], 1), cross_2nd_bracket_mat[1:-1, :, :]),
                                tf.expand_dims(brackets_p[1:-1], 2))
        As_d3 = 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets2[1:-1], 1), Bs[:-1, :, :] + Cs[1:, :, :]),
                              tf.expand_dims(brackets2[1:-1], 2))
        As_d4 = - 4 * tf.matmul(tf.matmul(tf.expand_dims(brackets2[1:-1], 1),
                                          (repeated_nodes_s2_small[:-2] - repeated_nodes_s2_small[1:-1]) * Bs[:-1, :,
                                                                                                           :] + (
                                          repeated_nodes_s2_small[2:] - repeated_nodes_s2_small[1:-1]) * Cs[1:, :, :]),
                                tf.expand_dims(brackets2[1:-1], 2))
        As_d = As_d1 + As_d2 + As_d3 + As_d4
        As_u1 = tf.matmul(tf.matmul(tf.expand_dims(brackets_pp[1:-2], 1), cross_2nd_bracket_mat[2:-1, :, :]),
                          tf.expand_dims(brackets[1:-2], 2))
        As_u2 = tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-2], 1), cross_2nd_bracket_mat[2:-1, :, :]),
                          tf.expand_dims(brackets_p[1:-2], 2))
        As_u3 = -2 * tf.matmul(tf.matmul(tf.expand_dims(brackets2[1:-2], 1), Bs[1:-1, :, :]),
                               tf.expand_dims(brackets2[1:-2], 2))
        As_u4 = 4 * (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p2[1:-2], 1), Bs[1:-1, :, :]), tf.expand_dims(brackets2[1:-2], 2))
        As_u = As_u1 + As_u2 + As_u3 + As_u4
        rhs1 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(repeated_bracket_pp, 4), Q1s[:, :, :, 1:, :, :] - Q1s[:, :, :, :-1, :, :]),
            tf.expand_dims(repeated_bracket, 5))
        rhs2 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(repeated_bracket_p, 4), Q1s[:, :, :, 1:, :, :] - Q1s[:, :, :, :-1, :, :]),
            tf.expand_dims(repeated_bracket_p, 5))
        rhs3 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(repeated_bracket2, 4), Q2s[:, :, :, :-1, :, :] - Q2s[:, :, :, 1:, :, :]),
            tf.expand_dims(repeated_bracket2, 5))
        rhs4 = 4 * tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket_p2, 4),
                                       (repeated_nodes_s2[:, :, :, 2:] - repeated_nodes_s2[:, :, :, 1:-1]) * Q2s[:, :,
                                                                                                             :, 1:, :,
                                                                                                             :] - (
                                       repeated_nodes_s2[:, :, :, :-2] - repeated_nodes_s2[:, :, :, 1:-1]) * Q2s[:, :,
                                                                                                             :, :-1, :,
                                                                                                             :]),
                             tf.expand_dims(repeated_bracket2, 5))
        rhs = rhs1 + rhs2 + rhs3 + rhs4
    else:
        As_l1 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[2:-1], 1), Bs[1:-1, :, :] + 2 * cross_bracket_mat[1:-2, :, :]),
            tf.expand_dims(brackets[2:-1], 2))
        As_l2 = 2 * (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_pp[2:-1], 1), cross_bracket_mat[1:-2, :, :]),
            tf.expand_dims(brackets[2:-1], 2))
        As_l3 = 2 * (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[2:-1], 1), cross_bracket_mat[1:-2, :, :]),
            tf.expand_dims(brackets_p[2:-1], 2))
        As_l = As_l1 + As_l2 + As_l3
        As_d1 = - 2 * tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-1], 1),
                                          Cs[:-1, :, :] + Bs[1:, :, :] + 4 * cross_bracket_mat[1:-1, :, :]),
                                tf.expand_dims(brackets[1:-1], 2))
        As_d2 = tf.matmul(tf.matmul(tf.expand_dims(brackets_pp[1:-1], 1),
                                    (repeated_nodes_s1_small[2:] - repeated_nodes_s1_small[1:-1]) * Bs[1:, :, :] - (
                                    repeated_nodes_s1_small[1:-1] - repeated_nodes_s1_small[:-2]) * Cs[:-1, :, :]),
                          tf.expand_dims(brackets[1:-1], 2))
        As_d3 = tf.matmul(tf.matmul(tf.expand_dims(brackets_p[1:-1], 1),
                                    (repeated_nodes_s1_small[2:] - repeated_nodes_s1_small[1:-1]) * Bs[1:, :, :] - (
                                    repeated_nodes_s1_small[1:-1] - repeated_nodes_s1_small[:-2]) * Cs[:-1, :, :]),
                          tf.expand_dims(brackets_p[1:-1], 2))
        As_d = As_d1 + As_d2 + As_d3
        As_u1 = 2 * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[1:-2], 1), 2 * cross_bracket_mat[2:-1, :, :] + Cs[1:-1, :, :]),
            tf.expand_dims(brackets[1:-2], 2))
        As_u2 = - 2 * (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_pp[1:-2], 1), cross_bracket_mat[2:-1, :, :]),
            tf.expand_dims(brackets[1:-2], 2))
        As_u3 = - 2 * (expanded_nodes[2:-1] - expanded_nodes[1:-2]) * tf.matmul(
            tf.matmul(tf.expand_dims(brackets_p[1:-2], 1), cross_bracket_mat[2:-1, :, :]),
            tf.expand_dims(brackets_p[1:-2], 2))
        As_u = As_u1 + As_u2 + As_u3
        rhs1 = 4 * tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket_p, 4),
                                       Q1s[:, :, :, :-1, :, :] - Q1s[:, :, :, 1:, :, :] + Q2s[:, :, :, 1:, :, :] - Q2s[
                                                                                                                   :, :,
                                                                                                                   :,
                                                                                                                   :-1,
                                                                                                                   :,
                                                                                                                   :]),
                             tf.expand_dims(repeated_bracket, 5))
        rhs2 = 2 * tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket_pp, 4),
                                       (repeated_nodes_s1[:, :, :, 2:] - repeated_nodes_s1[:, :, :, 1:-1]) * Q1s[:, :,
                                                                                                             :, 1:, :,
                                                                                                             :] - (
                                       repeated_nodes_s1[:, :, :, 1:-1] - repeated_nodes_s1[:, :, :, :-2]) * Q2s[:, :,
                                                                                                             :, :-1, :,
                                                                                                             :]),
                             tf.expand_dims(repeated_bracket, 5))
        rhs3 = 2 * tf.matmul(tf.matmul(tf.expand_dims(repeated_bracket_p, 4),
                                       (repeated_nodes_s1[:, :, :, 2:] - repeated_nodes_s1[:, :, :, 1:-1]) * Q1s[:, :,
                                                                                                             :, 1:, :,
                                                                                                             :] - (
                                       repeated_nodes_s1[:, :, :, 1:-1] - repeated_nodes_s1[:, :, :, :-2]) * Q2s[:, :,
                                                                                                             :, :-1, :,
                                                                                                             :]),
                             tf.expand_dims(repeated_bracket_p, 5))
        rhs = rhs1 + rhs2 + rhs3
    rhs = tf.reshape(rhs, [N, -1, L, tf.shape(rhs)[3]])
    As_l = tf.tile(tf.reshape(As_l, [1, 1, 1, tf.shape(As_l)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    As_d = tf.tile(tf.reshape(As_d, [1, 1, 1, tf.shape(As_d)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    As_u = tf.tile(tf.reshape(As_u, [1, 1, 1, tf.shape(As_u)[0]]), [N, tf.shape(Q1s)[1], L, 1])
    lam = tridiagonal_solve(As_l, As_d, As_u, rhs, n_intervals - 1)
    return lam


def bracket_2nd_prime_x(x, d):
    # x has shape [n]
    # gives back output of shape [n, d], out[i,:] = [0, 0, 1 * 2 * x[i]^0, ..., (d-2) * (d-1) * x[i]^(d-3)]
    # currently implemented in numpy for fixed nodes, has to be implemented in tensorflow for nodes to be learnt
    b = np.power(np.expand_dims(x, 1), np.concatenate((np.array([0, 0]), np.arange(d - 2)))) * np.arange(
        d) * np.concatenate((np.array([0]), np.arange(d - 1)))
    return (np.array(b, dtype='float32'))


def spline_2nd_differentiability_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals):
    # projects the piece-wise polynomials parametrized by Q1s and Q2s onto the space of 2nd differentiable piece-wise polynomials
    # cross_2nd__bracket_mat has shape [n_intervals+1, s1, s1], mat[i,:,:] corresponds to bracket_pp(node[i])*bracket(node[i])' + bracket_p(node[i])*bracket_p(node[i])' + bracket(node[i])*bracket_pp(node[i])'
    brackets = bracket_x(nodes, s1)
    brackets2 = bracket_x(nodes, s2)
    bracket_mat2 = tf.matmul(tf.expand_dims(brackets2, 2), tf.expand_dims(brackets2, 1))
    brackets_p = bracket_prime_x(nodes, s1)
    brackets_p2 = bracket_prime_x(nodes, s2)
    cross_bracket_mat = tf.matmul(tf.expand_dims(brackets, 2), tf.expand_dims(brackets_p, 1)) + tf.matmul(
        tf.expand_dims(brackets_p, 2), tf.expand_dims(brackets, 1))
    cross_bracket_mat2 = tf.matmul(tf.expand_dims(brackets2, 2), tf.expand_dims(brackets_p2, 1)) + tf.matmul(
        tf.expand_dims(brackets_p2, 2), tf.expand_dims(brackets2, 1))
    brackets_pp = bracket_2nd_prime_x(nodes, s1)
    cross_2nd_bracket_mat = tf.matmul(tf.expand_dims(brackets, 2), tf.expand_dims(brackets_pp, 1)) + 2 * tf.matmul(
        tf.expand_dims(brackets_p, 2), tf.expand_dims(brackets_p, 1)) + tf.matmul(tf.expand_dims(brackets_pp, 2),
                                                                                  tf.expand_dims(brackets, 1))
    repeated_cross_bracket_mat = tf.tile(tf.reshape(cross_bracket_mat, [1, 1, 1, tf.shape(cross_bracket_mat)[0],
                                                                        tf.shape(cross_bracket_mat)[1],
                                                                        tf.shape(cross_bracket_mat)[2]]),
                                         [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    repeated_cross_2nd_bracket_mat = tf.tile(tf.reshape(cross_2nd_bracket_mat,
                                                        [1, 1, 1, tf.shape(cross_2nd_bracket_mat)[0],
                                                         tf.shape(cross_2nd_bracket_mat)[1],
                                                         tf.shape(cross_2nd_bracket_mat)[2]]),
                                             [N, tf.shape(Q1s)[1], L, 1, 1, 1])
    repeated_nodes_s2_small = np.tile(np.reshape(nodes, [-1, 1, 1]), [1, s2, s2])
    repeated_nodes_s1_small = np.tile(np.reshape(nodes, [-1, 1, 1]), [1, s1, s1])
    if p % 2 == 0:
        Bs = -bracket_mat2[1:, :, :] + (
                                       repeated_nodes_s2_small[:-1] - repeated_nodes_s2_small[1:]) * cross_bracket_mat2[
                                                                                                     1:, :, :]
        Cs = -bracket_mat2[:-1, :, :] + (repeated_nodes_s2_small[1:] - repeated_nodes_s2_small[
                                                                       :-1]) * cross_bracket_mat2[:-1, :, :]
        lam = lambda_2nd_differentiability(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals, brackets, brackets2,
                                           brackets_p, brackets_p2, brackets_pp, cross_bracket_mat,
                                           cross_2nd_bracket_mat, Bs, Cs)
        expanded_lam_s1 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s1, s1])
        expanded_lam_s2 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s2, s2])
        repeated_Bs = tf.tile(tf.reshape(Bs, [1, 1, 1, np.shape(Bs)[0], np.shape(Bs)[1], np.shape(Bs)[2]]),
                              [N, tf.shape(Q1s)[1], L, 1, 1, 1])
        repeated_Cs = tf.tile(tf.reshape(Cs, [1, 1, 1, np.shape(Cs)[0], np.shape(Cs)[1], np.shape(Cs)[2]]),
                              [N, tf.shape(Q1s)[1], L, 1, 1, 1])
        Q1s_new = Q1s - expanded_lam_s1[:, :, :, 1:] / 2 * repeated_cross_2nd_bracket_mat[:, :, :, 1:, :,
                                                           :] + expanded_lam_s1[:, :, :,
                                                                :-1] / 2 * repeated_cross_2nd_bracket_mat[:, :, :, :-1,
                                                                           :, :]
        Q2s_new = Q2s - expanded_lam_s2[:, :, :, 1:] * repeated_Bs + expanded_lam_s2[:, :, :, :-1] * repeated_Cs
    else:
        Bs = 2 * cross_bracket_mat[:-1, :, :] - (repeated_nodes_s1_small[1:] - repeated_nodes_s1_small[
                                                                               :-1]) * cross_2nd_bracket_mat[:-1, :, :]
        Cs = 2 * cross_bracket_mat[1:, :, :] + (repeated_nodes_s1_small[1:] - repeated_nodes_s1_small[
                                                                              :-1]) * cross_2nd_bracket_mat[1:, :, :]
        lam = lambda_2nd_differentiability(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals, brackets, brackets2,
                                           brackets_p, brackets_p2, brackets_pp, cross_bracket_mat,
                                           cross_2nd_bracket_mat, Bs, Cs)
        expanded_lam_s1 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s1, s1])
        expanded_lam_s2 = tf.tile(tf.expand_dims(tf.expand_dims(lam, -1), -1), [1, 1, 1, 1, s2, s2])
        repeated_Bs = tf.tile(tf.reshape(Bs, [1, 1, 1, tf.shape(Bs)[0], tf.shape(Bs)[1], tf.shape(Bs)[2]]),
                              [N, tf.shape(Q1s)[1], L, 1, 1, 1])
        repeated_Cs = tf.tile(tf.reshape(Cs, [1, 1, 1, tf.shape(Cs)[0], tf.shape(Cs)[1], tf.shape(Cs)[2]]),
                              [N, tf.shape(Q1s)[1], L, 1, 1, 1])
        Q1s_new = Q1s + expanded_lam_s1[:, :, :, 1:] * repeated_cross_bracket_mat[:, :, :, 1:, :, :] - expanded_lam_s1[
                                                                                                       :, :, :,
                                                                                                       :-1] / 2 * repeated_Bs
        Q2s_new = Q2s - expanded_lam_s2[:, :, :, 1:] / 2 * repeated_Cs + expanded_lam_s2[:, :, :,
                                                                         :-1] * repeated_cross_bracket_mat[:, :, :, :-1,
                                                                                :, :]
    return Q1s_new, Q2s_new


def alternating_projections(Q1s, Q2s, steps, nodes, n_intervals, p, s1, s2, N, L, smooth):
    n_constraints = smooth + 1
    for step in xrange(steps):
        mod = step % n_constraints
        if mod == 0:  # SPD projection
            Q1s = spline_spd_proj(Q1s, s1)
            Q2s = spline_spd_proj(Q2s, s2)
        if mod == 1:  # continuity projection projection
            Q1s, Q2s = spline_continuity_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals)
        if mod == 2:  # differentiability projection projection
            Q1s, Q2s = spline_differentiability_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals)
        if mod == 3:  # second differentiability projection
            Q1s, Q2s = spline_2nd_differentiability_proj(Q1s, Q2s, nodes, p, s1, s2, N, L, n_intervals)
    return Q1s, Q2s


def f_nn(hid, units, steps, nodes, n_intervals, p, s1, s2, N, L, smooth):
    with tf.variable_scope("f"):
        for i in xrange(len(units) - 1):
            hid = tf.layers.dense(inputs=hid, units=units[i], activation=tf.nn.relu)
        hid = tf.layers.dense(inputs=hid, units=units[-1], activation=tf.identity)
        Q1s, Q2s = reshape_for_params(hid, s1, s2, N, L, n_intervals)
        Q1s, Q2s = alternating_projections(Q1s, Q2s, steps, nodes, n_intervals, p, s1, s2, N, L,
                                           smooth)  # maps parameters onto feasible space
    return Q1s, Q2s


def which_indices(Y, nodes, T):
    # Y has shape [N, None, L, max_lengths], so each Y[n, b, l, :] is a spike train.
    # nodes has shape [n_nodes]
    # output ind has the same shape as Y, and ind[n, b, l, t] corresponds to the index in nodes of the lower node
    # output lower_nodes_at_ind has the same shape as Y and lower_nodes_at_ind[n, b, l, t] = nodes[ind[n, b, l, t]]
    # output upper_nodes_at_ind has the same shape as Y and upper_nodes_at_ind[n, b, l, t] = nodes[ind[n, b, l, t]+1]
    ind = tf.expand_dims(Y, 4) - nodes  # shape [N, None, L, max_lengths, n_nodes]
    ind = tf.where(tf.less(ind, 0.0), (T + 1.0) * tf.ones_like(ind), ind)
    ind = tf.argmin(ind, axis=4)
    lower_nodes_at_ind = tf.gather(nodes, ind)
    upper_ind = tf.where(tf.equal(lower_nodes_at_ind, T), ind, ind + 1)
    upper_nodes_at_ind = tf.gather(nodes, upper_ind)
    return ind, lower_nodes_at_ind, upper_nodes_at_ind


def Qs_at_indices(Q1s, Q2s, ind, max_lengths, n_intervals, s1, s2, N, L):
    # Q1s has shape [N,None,L,n_intervals,s1,s1] and Q2s has shape [N,None,L,n_intervals,s2,s2]
    # output1 has shape [N,None,L,max_lengths,s1,s1] and is such that output1[n,b,l,t,i,j] = Q1s[n,b,l,ind[n,b,l,t],i,j]
    # output2 has shape [N,None,L,max_lengths,s2,s2] and is such that output1[n,b,l,t,i,j] = Q2s[n,b,l,ind[n,b,l,t],i,j]
    Q1s_aux = tf.reshape(Q1s, [-1, n_intervals, s1, s1])
    ind_aux = tf.reshape(ind, [-1, max_lengths])
    res1 = tf.gather(Q1s_aux, ind_aux, axis=1)
    res1 = tf.transpose(tf.matrix_diag_part(tf.transpose(res1, [2, 3, 4, 0, 1])), [3, 0, 1, 2])
    res1 = tf.reshape(res1, [N, -1, L, max_lengths, s1, s1])
    Q2s_aux = tf.reshape(Q2s, [-1, n_intervals, s2, s2])
    res2 = tf.gather(Q2s_aux, ind_aux, axis=1)
    res2 = tf.transpose(tf.matrix_diag_part(tf.transpose(res2, [2, 3, 4, 0, 1])), [3, 0, 1, 2])
    res2 = tf.reshape(res2, [N, -1, L, max_lengths, s2, s2])
    return res1, res2


def bracket_tf(Y, d):
    # Y has shape [N, None, L, max_lengths]
    # gives back a tensor of shape [N, None, L, max_lengths, d+1] where out[n,b,l,t,j] = Y[n,b,l,t]^j
    bracket = []
    for i in xrange(d + 1):
        bracket.append(tf.pow(Y, i))
    bracket = tf.transpose(bracket, [1, 2, 3, 4, 0])
    return bracket


def eval_pol(Q1s, Q2s, Y, p, s1, s2, lower_nodes, upper_nodes):
    # Q1s has shape [N,None,L,max_lengths,s1,s1], Q2s [N,None,L,max_lengths,s2,s2] and Y [N, None, L, max_lengths].
    # evaluates each of the N*None*L*max_lengths polynomials
    bracket_Y = bracket_tf(Y, s1 - 1)
    bracket2_Y = bracket_tf(Y, s2 - 1)
    res1 = tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(bracket_Y, 5), [0, 1, 2, 3, 5, 4]), Q1s),
                     tf.expand_dims(bracket_Y, 5))
    res1 = tf.squeeze(res1)
    res2 = tf.matmul(tf.matmul(tf.transpose(tf.expand_dims(bracket2_Y, 5), [0, 1, 2, 3, 5, 4]), Q2s),
                     tf.expand_dims(bracket2_Y, 5))
    res2 = tf.squeeze(res2)
    if p % 2 == 0:
        res2 = res2 * (Y - lower_nodes) * (upper_nodes - Y)
    else:
        res1 = res1 * (upper_nodes - Y)
        res2 = res2 * (Y - lower_nodes)
    return res1 + res2


def log_intensities(Y, Q1s, Q2s, nodes, mask, L, max_lengths, n_intervals, p, s1, s2, N, T, eps):
    ind, lower_nodes_at_ind, upper_nodes_at_ind = which_indices(Y, nodes, T)
    Q1s_at_ind, Q2s_at_ind = Qs_at_indices(Q1s, Q2s, ind, max_lengths, n_intervals, s1, s2, N, L)
    pol = eval_pol(Q1s_at_ind, Q2s_at_ind, Y, p, s1, s2, lower_nodes_at_ind, upper_nodes_at_ind)
    res = tf.log(tf.maximum(pol, eps)) * mask  # add eps to avoid taking log of 0
    return res


def powers(x, d):
    # gives back [x, x^2, ..., x^d, x^(d+1)], batched
    # should change so it generalizes for when x is a tensor
    bracket = []
    for i in xrange(d + 1):
        bracket.append(np.power(x, i + 1))
    bracket = np.array(bracket, dtype=np.float32)
    res = tf.transpose(bracket, [1, 0])
    return res


def pol_coeffs(Qs, s):
    # Qs has shape [N,None,L,n_intervals,s,s]
    # give back an output of shape [N,None,L,n_intervals,2s-1], where output[n,b,l,i,j] is the coefficient of the j-th
    # power in the polynomial bracket(x)'*Qs[n,b,l,i,:,:]*bracket(x)
    res = []
    for i in xrange(s):
        res.append(tf.reduce_sum(tf.matrix_diag_part(tf.reverse(Qs[:, :, :, :, :i + 1, :i + 1], [-1])), axis=4))
    for i in xrange(s - 1):
        res.append(tf.reduce_sum(tf.matrix_diag_part(tf.reverse(Qs[:, :, :, :, i + 1:, i + 1:], [-1])), axis=4))
    res = tf.transpose(res, [1, 2, 3, 4, 0])
    return res


def integrate_coeffs(coeffs, deg):
    # gives back the coefficients corresponding to the integral of the polynomials defined by coeffs
    res = []
    for i in xrange(deg + 1):
        res.append(coeffs[:, :, :, :, i] / (i + 1))
    res = tf.transpose(res, [1, 2, 3, 4, 0])
    return res


def shift_coeffs(coeffs, deg):
    # gives back the coefficients corresponding to multiplying the polynomials defined by coeffs by x
    res = []
    for i in xrange(deg + 2):
        if i == 0:
            res.append(tf.zeros_like(coeffs[:, :, :, :, 0]))
        else:
            res.append(coeffs[:, :, :, :, i - 1])
    res = tf.transpose(res, [1, 2, 3, 4, 0])
    return res


def integral(Q1s, Q2s, nodes, p, N, L, s1, s2):
    # computes the integral of each spline defined by Q1s [N,None,L,n_intervals,s,s] and Q2s [N,None,L,n_intervals]
    coeffs_1 = pol_coeffs(Q1s, s1)
    lower_nodes_powers = powers(nodes[:-1], p)
    upper_nodes_powers = powers(nodes[1:], p)
    coeffs_2 = pol_coeffs(Q2s, s2)
    if p % 2 == 0:
        int_coeffs_1 = integrate_coeffs(coeffs_1, p)  # corresponds to the [x]'Q1[x] term
        integrals_1 = int_coeffs_1 * (upper_nodes_powers - lower_nodes_powers)
        integrals_1 = tf.reduce_sum(integrals_1, axis=4)
        coeffs_2_1 = - shift_coeffs(shift_coeffs(coeffs_2, p - 2), p - 1)
        int_coeffs_2_1 = integrate_coeffs(coeffs_2_1, p)  # corresponds to the x^2 * [x]'Q2[x] term
        integrals_2_1 = int_coeffs_2_1 * (upper_nodes_powers - lower_nodes_powers)
        integrals_2_1 = tf.reduce_sum(integrals_2_1, axis=4)
        coeffs_2_2 = shift_coeffs(coeffs_2, p - 2)
        nodes_2_2 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1]), [N, tf.shape(Q1s)[1], L, 1, tf.shape(coeffs_2_2)[4]])
        coeffs_2_2 = coeffs_2_2 * (nodes_2_2[:, :, :, :-1] + nodes_2_2[:, :, :, 1:])
        int_coeffs_2_2 = integrate_coeffs(coeffs_2_2, p - 1)  # corresponds to the x * [x]'Q2[x] term
        integrals_2_2 = int_coeffs_2_2 * (upper_nodes_powers[:, :-1] - lower_nodes_powers[:, :-1])
        integrals_2_2 = tf.reduce_sum(integrals_2_2, axis=4)
        nodes_2_3 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1]), [N, tf.shape(Q1s)[1], L, 1, tf.shape(coeffs_2)[4]])
        coeffs_2_3 = - coeffs_2 * (nodes_2_3[:, :, :, :-1] * nodes_2_3[:, :, :, 1:])
        int_coeffs_2_3 = integrate_coeffs(coeffs_2_3, p - 2)  # corresponds to the [x]'Q2[x] term
        integrals_2_3 = int_coeffs_2_3 * (upper_nodes_powers[:, :-2] - lower_nodes_powers[:, :-2])
        integrals_2_3 = tf.reduce_sum(integrals_2_3, axis=4)
        integrals_2 = integrals_2_1 + integrals_2_2 + integrals_2_3
    else:
        coeffs_1_1 = - shift_coeffs(coeffs_1, p - 1)
        int_coeffs_1_1 = integrate_coeffs(coeffs_1_1, p)  # corresponds to the x * [x]'Q1[x] term
        integrals_1_1 = int_coeffs_1_1 * (upper_nodes_powers - lower_nodes_powers)
        integrals_1_1 = tf.reduce_sum(integrals_1_1, axis=4)
        nodes_1_2 = tf.tile(tf.reshape(nodes, [1, 1, 1, -1, 1]), [N, tf.shape(Q1s)[1], L, 1, tf.shape(coeffs_1)[4]])
        coeffs_1_2 = coeffs_1 * nodes_1_2[:, :, :, 1:]
        int_coeffs_1_2 = integrate_coeffs(coeffs_1_2, p - 1)  # corresponds to the [x]'Q1[x] term
        integrals_1_2 = int_coeffs_1_2 * (upper_nodes_powers[:, :-1] - lower_nodes_powers[:, :-1])
        integrals_1_2 = tf.reduce_sum(integrals_1_2, axis=4)
        integrals_1 = integrals_1_1 + integrals_1_2
        coeffs_2_1 = shift_coeffs(coeffs_2, p - 1)
        int_coeffs_2_1 = integrate_coeffs(coeffs_2_1, p)  # corresponds to the x * [x]'Q2[x] term
        integrals_2_1 = int_coeffs_2_1 * (upper_nodes_powers - lower_nodes_powers)
        integrals_2_1 = tf.reduce_sum(integrals_2_1, axis=4)
        coeffs_2_2 = - coeffs_2 * nodes_1_2[:, :, :, :-1]
        int_coeffs_2_2 = integrate_coeffs(coeffs_2_2, p - 1)  # corresponds to the [x]'Q2[x] term
        integrals_2_2 = int_coeffs_2_2 * (upper_nodes_powers[:, :-1] - lower_nodes_powers[:, :-1])
        integrals_2_2 = tf.reduce_sum(integrals_2_2, axis=4)
        integrals_2 = integrals_2_1 + integrals_2_2
    return integrals_1 + integrals_2