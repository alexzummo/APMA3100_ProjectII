# Created by Alexander W. Zummo and Sophie Meyer
# runs simulations of the package drops described in the project
# the RNG class is used for generating random numbers
# numpy used for means
# scipy used for normal distribution and CDF tools
# matplotlib used to create graphs

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class RNG:
    def __init__(self, x_0, a, c, k):
        self.seed = x_0
        self.multiplier = a
        self.increment = c
        self.modulus = k
        self.nums = []

    def generate(self, n=1000):
        """
        Generates n random numbers and returns as a list
        :param n: number of random numbers to generate
        :return: a list of random numbers of length n
        """
        ret = []
        x_i = self.seed
        for i in range(n):
            x_i = (x_i * self.multiplier + self.increment) % self.modulus  # x_i = (x_i-1 * a + c) % K
            ret.append(x_i / self.modulus)  # u_i = x_i / K
        self.nums = ret
        return ret

    def show_51(self):
        "returns u_51, u_52, and u_53. Used for the report"
        nums = self.generate(55)
        return nums[50], nums[51], nums[52]


class CDF:
    def __init__(self, observations):
        self.data = observations
        self.data.sort()
        self.size = len(self.data)

    def plt(self, x):
        """returns Probability that an observation is Less Than (PLT) the parameter x"""
        inc_p = 1 / self.size  # incremental probability, I.E. 1 / num observations
        p = 0  # total probability to return
        # i = 0
        for j in range(self.size):
            i = self.data[j]
            if i < x:
                p += inc_p # for each observation less than x, increase P[z < X]
            else:
                break
        # print("value:", x, "was less than", i)
        return p

    def get_graphable(self):
        """returns a tuple: first list is x axis, second list is y axis"""
        x_axis = []
        y_axis = []
        inc_p = 1 / self.size  # incremental probability
        total = 0
        for j in range(self.size):
            total += inc_p
            x_axis.append(total)
            y_axis.append(self.data[j])
        return x_axis, y_axis


def map_rand(x):
    """maps the randomly generated number to a value for the radius"""
    t = 57
    a = 1/t
    return ((-2 * math.log(1-x)) // a**2) ** (1/2)  # this is the inverse cdf


def simulate_package_drop(queue, size):
    """runs a simulation of the package dropping process and returns the sample mean"""
    total = 0.0
    # histogram = []  # I used this to collect data that verified that this method would produce the correct pdf
    for i in range(size):
        x = queue.pop(0)
        total += map_rand(x)
        # histogram.append(mapped)
    # print(histogram)
    return total / size


def convert_to_z(x, n, mean, std_dev):
    """return z score given parameter x of normal dist with parameters n, mean, and std_dev"""
    num = (x - mean)
    den = (std_dev / (n ** .5))
    return num / den


rng = RNG(1000, 24693, 3967, 2**17)
rand_queue = rng.generate(229900)  # I'm using a queue so that each random number is used exactly once
sample_sizes = [10, 30, 50, 100, 150, 250, 500, 1000]

means = {}
for size in sample_sizes:
    means[size] = []
    for i in range(110):
        mean = simulate_package_drop(rand_queue, size)
        means[size].append(mean)

mean_x = 57 * ((math.pi / 2) ** .5)  # theoretical overall mean
std_dev_x = ((4 - math.pi) / (2 * ((1 / 57) ** 2))) ** .5  # theoretical overall standard distribution

z_conv = {}  # dict with keys as n values and values as lists of z scores

for size in sample_sizes:
    k_i = means[size]
    z_conv[size] = []  # initialize list
    for m in k_i:
        z_i = convert_to_z(m, size, mean_x, std_dev_x)  # convert distances to z scores
        z_conv[size].append(z_i)

graphable = {}  # keys as n values, values as [[110 cdf probs], [110 z scores], [7 absolute differences]]
diffs = {}  # keys as n values, values as lists of the 7 absolute differences
j_vals = [-1.4, -1.0, -0.5, 0, 0.5, 1.0, 1.4]
for n in z_conv.keys():
    graphable[n] = []
    cdf = CDF(z_conv[n])
    temp = cdf.get_graphable()
    graphable[n] = [temp[0], temp[1]]
    diffs[n] = []
    graphable[n].append([])
    for j in j_vals:
        fn_zj = cdf.plt(j)  # use CDF object to get P[z<Z]
        ncdf = norm.cdf(j)  # use scipy to get phi(z)
        diffs[n].append(abs(fn_zj - ncdf))
        graphable[n][2].append(fn_zj)  # I used graphable[2] to store the 7 CDF values

f = open("clt_results.txt", 'w')
print("Printing results to file...")
for n in diffs.keys():
    print("n =", n, file=f)
    print("MAD:", max(diffs[n]), file=f)
    for i in diffs[n]:
        print(i, file=f)
    print("=" * 40, file=f)
print("Done")

print("Creating graphs...")
for n in sample_sizes:
    print("n =", n)
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 2.5*sigma, mu + 2.5*sigma, 100)
    y = norm.cdf(x, mu, sigma)
    plt.plot(x, y)  # plot CDF of normal distribution
    plt.plot(j_vals, graphable[n][2], marker='.', linestyle='None')  # plot 7 CDF values
    max_diff = max(diffs[n])
    j_ind = diffs[n].index(max_diff)
    x = j_vals[j_ind]
    y = graphable[n][2][j_ind]
    plt.axvspan(x-.01, x+.01 , color='red', alpha=0.5)
    plt.plot(x, y, marker='o', color='red')  # Highlight CDF value corresponding to the MAD
    plt.ylabel("cumulative probability")
    plt.xlabel("z value")
    plt.title("n = " + str(n))
    red_patch = mpatches.Patch(color='red', label='MADn')
    blue_patch = mpatches.Patch(color='blue', label='Normal CDF')
    orange_patch = mpatches.Patch(color='orange', label='Empirical CDF')
    plt.legend(handles=[red_patch, blue_patch, orange_patch])
    plt.savefig("n"+str(n)+".png")
    plt.clf()  # clear plot to start over with next
print("Done")
