from typing import Any, Callable, Optional

import h5py
import torch

from multiprocessing import Queue
# from threading import Semaphore, Thread


class Stats():
    def __init__(self, miss_era: float) -> None:
        # The exponential running average miss rate (alpha = Cache.alpha)
        self.miss_era = miss_era


class Cache():
    def __init__(self, file: str) -> None:
        self.file = file
        self.h5: Optional[h5py.File] = None
        self.alpha = 0.1
        self.stats = Stats(miss_era=0)
        self.queue = Queue(maxsize=50)
        self.exit_worker = False

    #def _worker(self, init_sem: Semaphore):
    #    print("[Cache-Worker] Entered", flush=True)
    #    self.exit_worker = False
    #    with h5py.File(self.file, mode='a', libver='latest') as h5:
    #        h5.swmr_mode = True
    #        print("[Cache-Worker] Init Done", flush=True)
    #        init_sem.release()
    #        while not self.exit_worker:
    #            task = self.queue.get(block=True)
    #            if task is not None:
    #                key, element = task
    #                h5.create_dataset(key, data=element)

    def __enter__(self) -> "Cache":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        assert self.h5 is not None
        self.close()

    def open(self) -> None:
        assert self.h5 is None
        print("[Cache] Opening", flush=True)
        #init_sem = Semaphore(0)
        #self.worker = Thread(target=self._worker, args=(init_sem,), daemon=True)
        #self.worker.start()
        #print("[Cache] Waiting for Worker", flush=True)
        #init_sem.acquire()
        #self.h5 = h5py.File(self.file, mode="r", swmr=True, libver='latest')
        self.h5 = h5py.File(self.file, mode="a", libver='latest', locking=False)
        print("[Cache] Init done", flush=True)

    def close(self) -> None:
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
        self.exit_worker = True
        #self.queue.put(None)
        #self.worker.join()
        #self.worker = None
        print("[Cache] Closed")

    def __contains__(self, key) -> bool:
        return key in self.h5

    def get_or_add(self, key, generator: Callable, *args, **kwargs) -> Any:
        assert self.h5 is not None
        if key in self.h5:
            self.stats.miss_era *= (1-self.alpha)
            return torch.from_numpy(self.h5[key][0])
        else:
            self.stats.miss_era = self.alpha + (1-self.alpha)*self.stats.miss_era
            element = generator(*args, **kwargs)
            self.h5.create_dataset(key, data=[element.cpu().numpy()])
            #self.queue.put((key, [element.cpu().numpy()]))
            return element
