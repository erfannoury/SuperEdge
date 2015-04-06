import numpy as np
from bsds500 import load_data
from datetime import datetime
def main():
    now = datetime.now()
    data = load_data(['train', 'val', 'test'])
    print 'loading all data took ', (datetime.now() - now)


if __name__ == '__main__':
    main()
