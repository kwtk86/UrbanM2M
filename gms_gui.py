# @Time : 2023/12/11 16:59
# @Author : 8
# @File : gms_gui.py
"""

"""
from m2m_core.gms_interface import gms_m2m_main

gms_m2m_main('./data-yrd',
             2006,
             2012,
             6,
             './trained_models/convlstm-t08_14_23-e24.pth',
             './data-yrd/prob',
             './data-yrd/sim',
             [500000,500000,500000,500000,500000,500000],
             8,
             8)
