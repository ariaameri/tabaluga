from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os
import cv2
import pathlib
import numpy as np

def task_normal(x):

    out = np.array([
        cv2.resize(
            image,
            dsize=(512, 512)
        )
        for image
        in x
    ])

    out = out / 255.

    return out

def task_simple_map(x):

    out = map(resizer, x)

    out = map(normalizer, out)

    return np.array(list(out))

def resizer(x):
    return cv2.resize(x, (512, 512))

def normalizer(x):
    return x / 255

def task_multi(x):

    with ProcessPoolExecutor() as exec:

        out = exec.map(resizer, x)

        out = exec.map(normalizer, out)

    return np.array(list(out))

def task_thread(x):

    with ThreadPoolExecutor() as exec:

        out = exec.map(resizer, x)

        out = exec.map(normalizer, out)

    return np.array(list(out))

def main():

    img_paths = [item.as_posix() for item in pathlib.Path('/Users/aria/Downloads/Nexus_Week23_Part2_A_2020_Completed/img').iterdir()]
    imgs = np.array([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_paths])
    # print(imgs.shape)

    print(task_normal(imgs).shape)

    # print(task_simple_map(imgs).shape)
    #
    # print(task_multi(imgs).shape)
    #
    # print(task_thread(imgs).shape)

    # executor = ProcessPoolExecutor(max_workers=3)
    # list(executor.map(task, range(10)))
    # task1 = executor.submit(task)
    # task2 = executor.submit(task)

if __name__ == '__main__':
    # main()

    from nexdeepml.util.panacea import Panacea

    def alaki(x):

        return x * 3

    Panacea({}).update({}, {'alaki': np.ones(70)}).update({},
                                                          {'$map_on_value_process': {'alaki': alaki}}).print()