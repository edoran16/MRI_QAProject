import pyxnat
from pyxnat import Interface

interface = Interface(server='https://130.209.143.85/app/template/Login.vm#!', user='emma',
                       password='Medphys_0520')
print(interface)
interface.select.projects().get()

