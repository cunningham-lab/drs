import numpy as np

def ker_mat(ker, xs, sigma2):
    #given a kernel ker and points xs, computes the kernel matrix
    n = len(xs)
    K = np.zeros([n,n])
    for i in xrange(n):
        for j in xrange(n):
            K[i,j] = ker(xs[i], xs[j], sigma2)
    return(K)

def integrate_piece_linear(xs, ys):
    #gives back the integral of the piece wise linear function defined by xs and ys
    #xs should be ordered in increasing order
    bases = xs[1:] - xs[:-1]
    heights = (ys[1:] + ys[:-1]) / 2.0
    integral = np.sum(bases * heights)
    return(integral)

def unif_sampler(xs, ys):
    #use rejection sampling to sample from the distribution proportional to the piece-wise linear function defined by
    #xs and ys.
    #xs should be ordered in increasing order and start at 0
    b = 0
    M = np.max(ys)
    T = xs[-1]
    while b==0:
        u = T * np.random.rand(1)
        upper_ind = np.argmax(xs > u)
        lower_ind = upper_ind - 1
        y_at_u = ys[lower_ind] + (u - xs[lower_ind]) * (ys[upper_ind] - ys[lower_ind]) / (xs[upper_ind] - xs[lower_ind])
        v = M * np.random.rand(1)
        if v < y_at_u:
            b = 1
    return(u)

def pp_sample(xs, ys, sampler):
    #gives back a sample from a Poisson Process whose intensity is a piece-wise linear function defined by xs and ys
    #the sample is ordered in increasing order
    s = []
    N = np.random.poisson(integrate_piece_linear(xs, ys))
    for i in xrange(N):
        s.append(np.float(sampler(xs, ys)))
    return(s)

def create_data_set(xs, N, types_of_trials, samples_per_type, mu, sigma2, ker):
    K = ker_mat(ker, xs, sigma2)
    R = types_of_trials * samples_per_type
    true_ys = np.zeros([N, types_of_trials, len(xs)])
    ys = np.zeros([N, R, len(xs)])
    samples = []
    max_lengths = 0
    lengths = np.zeros([N,R], dtype="int")
    #generate the true intensity functions for each type of trial
    for n in xrange(N):
        for r_type in xrange(types_of_trials):
            true_ys[n,r_type,:] = np.exp(np.random.multivariate_normal(mu, K))
    #sample the PPs
    for n in xrange(N):
        s = []
        for r in xrange(R):
            r_type = r % types_of_trials
            ys[n,r,:] = true_ys[n,r_type,:]
            sample = pp_sample(xs, ys[n,r,:], unif_sampler)
            s.append(sample)
            lengths[n,r] = len(sample)
            if len(sample) > max_lengths:
                max_lengths = len(sample)
        samples.append(s)
    #transform the samples into a numpy array
    spike_trains = np.zeros([N, R, max_lengths])
    masks = np.zeros([N, R, max_lengths])
    for n in xrange(N):
        for r in xrange(R):
            l = len(samples[n][r])
            spike_trains[n,r,:l] = np.sort(samples[n][r])
            masks[n,r,:l] = 1.0
    return(spike_trains, masks, lengths, max_lengths, ys)

def RBF(t1, t2, sigma2):
    ans = np.exp(-(t1 - t2)**2 / (2.0 * sigma2))
    return(ans)