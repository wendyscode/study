import subprocess
import sys
import time
import os

base_dir = r'C:\study\keras2'   # 6개 파일이 있는 폴더

files = [
    'keras73_vgg16_13_cifar10.py',
    'keras73_vgg16_14_cifar100.py',
    'keras73_vgg16_15_man_horse.py',
    'keras73_vgg16_16_rps.py',
    'keras73_vgg16_17_man_woman.py',
    'keras73_vgg16_18_cat_dog.py',
]

for f in files:
    path = os.path.join(base_dir, f)
    print(f"\n########## {f} 시작 ##########")

    if not os.path.exists(path):
        print(f"파일 없음: {path}")
        continue

    start = time.time()
    result = subprocess.run([sys.executable, '-u', path])

    status = "성공" if result.returncode == 0 else "에러"
    print(f"########## {f} {status} ({(time.time()-start)/60:.1f}분) ##########")