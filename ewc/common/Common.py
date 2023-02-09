# -*- coding: utf-8 -*-
# Author : JunHyuck Kim
# e-mail : junhyuck.kim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.

from pycmmn.Singleton import Singleton
from ewc.common.Constants import Constants

class Common(object, metaclass=Singleton):
    print(Constants.CONSTANT_VARIABLE)
    print(Constants.MODEL_RESOURCE_PATH)

if __name__ == '__main__':
    Common()