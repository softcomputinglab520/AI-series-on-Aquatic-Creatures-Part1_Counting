from .videotoimg import filename
import os

def rename(out, date, times):
    files = [x for x in os.listdir(out)]
    amount = len(files)
    for i in range(amount):
        new_name = filename(i + 1, len(files))
        os.rename(out + files[i], out + date + times + '_' + new_name + '.jpg')
