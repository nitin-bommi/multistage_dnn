
from zipfile import ZipFile

with ZipFile('data/ChestXRay2017.zip', 'r') as z:
    z.extractall('data/pneumonia')
