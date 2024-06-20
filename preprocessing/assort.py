from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

labels = pd.read_csv(r'c:\Users\mishr\Desktop\flowers\archive\labels.csv')

train_dir =r'C:\Users\mishr\Desktop\flowers\archive\files'
DR = r"C:\Users\mishr\Desktop\flowers\archive\flowers"
if not os.path.exists(DR):
    os.mkdir(DR)

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(DR + str(class_name)):
        os.mkdir(DR + str(class_name))
    src_path = str(train_dir) + '/'+ str(filename) + '.jpg'
    dst_path = DR + str(class_name) + '/' + str(filename)
    try:
        shutil.move(src_path, dst_path)
        print("sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))