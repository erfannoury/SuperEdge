from poisson_disc import PoissonDiskSampler
import matplotlib.pyplot as plt
from mpltools import style
style.use(['ggplot'])
from datetime import datetime

def main():
    width = 480
    height = 320
    radius = 5
    now = datetime.now()
    pds = PoissonDiskSampler(width, height, radius)
    samples = pds.get_sample()
    print 'samples size: ', len(samples)
    print 'generating samples took ', (datetime.now() - now)
    xs = [int(s[0]) for s in samples]
    ys = [int(s[1]) for s in samples]
    plt.plot(xs, ys, 'o')
    plt.xlim([0,width])
    plt.ylim([0,height])
    plt.show()



if __name__ == '__main__':
    main()