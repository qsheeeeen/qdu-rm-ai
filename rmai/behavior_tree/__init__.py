import xml.etree.ElementTree as ETree

from .action import *
from .control import *
from .decorator import *


def parse(xml_dir):
    tree = ETree.parse(xml_dir)
    root = tree.getroot()
