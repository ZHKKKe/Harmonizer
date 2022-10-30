from . import logger


def dict_value(dictionary, name, default=None, err=False):
    if dictionary is None:
        if err:
            logger.log_err('The given dictionary is None\n')
        else:
            return default

    if name in dictionary:
        return dictionary[name]
    elif err:
        logger.log_err('Cannot find key: {0}'.format(name))
    else:
        return default
