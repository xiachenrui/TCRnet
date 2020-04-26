import sys
import os
# print (sys.argv[0])
# print(os.path.abspath('.'))
path =os.path.abspath(os.curdir)
package = os.path.join(path,'deeptcr')
# if path == 'E:\OneDrive\program\demo\deeptcr':
#     print("ture")
# else:
#     print("flase")
sys.path.append(package)
# from utils import read_data_train , read_data_test
