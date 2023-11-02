import os

try:
    os.remove(os.path.join(os.getcwd(), 'instructions/_latest'))
except:
    pass
os.symlink(os.path.join(os.getcwd(), f'instructions/main'), os.path.join(os.getcwd(), 'instructions/_latest'))