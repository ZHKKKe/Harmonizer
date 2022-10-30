import logging


# config logging here
format_str = '%(message)s'
formatter = logging.Formatter(format_str)
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger('TorchTask')

# ---------------------------------------------------------------------
# Functions for logging
# ---------------------------------------------------------------------

def log_mode(debug=False):
    global logger

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def log_file(fpath, debug=False):
    global logger
    global formatter

    fh = logging.FileHandler(fpath)
    if debug:
        fh.setLevel(logging.DEBUG)
    else:
        fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)


def log_info(message):
    global logger
    
    out = message
    if isinstance(message, list):
        out = ''.join(message)
    
    logger.info(out)


def log_warn(message):
    global logger

    out = message
    if isinstance(message, list):
        out = ''.join(message)
    out = '\n' + '=' * 36 + ' WARN ' + '=' * 36 + '\n' + out + '=' * 78 + '\n'
                                                    
    logger.warn(out)


def log_err(message):
    global logger

    out = message
    if isinstance(message, list):
        out = ''.join(message)
    out = '\n' + '=' * 35 + ' ERROR ' + '=' * 36 + '\n' + out + '=' * 78 + '\n'

    logger.error(out)
    exit()


class AvgMeter:
    """ Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format)


class AvgMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def keys(self):
        return self.meters.keys()

    def has_key(self, key):
        return key in self.meters.keys()

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AvgMeter()
        self.meters[name].update(value, n)

    def reset(self, name=None):
        if name is None:
            for meter in self.meters.values():
                meter.reset()
        elif name in self.meters.keys():
            self.meters[name].reset()
        else:
            log_err('Unknown key value for AvgMeterSet: {0}\n'.format(name)) 

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
