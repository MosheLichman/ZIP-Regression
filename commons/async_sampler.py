"""
Asynchronous version of np.random.choice().

When the number of elements you want to choose from is in the millions, the sampling can take up to minutes and
completely dominate the running time of the code. To avoid that, I implemented an async sampler that make samples in the
background and just hands them out on request.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import atexit
from commons import log_utils as log
from multiprocessing import Queue, Process


class AsyncSampler(object):
    def __init__(self, num_proc=1, queue_size=3):
        """Instantiates an AsyncSampler object.

         Args
        ------
            1. num_proc:    <int>   number of process to use to produce the samples (default = 1)
            2. queue_size:  <int>   how many samples to save in the background (default = 3)
        """
        self.proc_pools = {}
        self.samplers = {}
        
        self.num_proc = num_proc
        self.q_size = queue_size

        # Register the stop_sampler method to the destructor so it will free all the memory and stop zombie threads.
        atexit.register(self.stop_sampler)
        
    @staticmethod
    def _async_sampler(queue, num_points, batch_size):
        """Runs np.random.choice(num_points, batch_size) and stores it in the queue. """
        while True:
            idx = np.random.choice(num_points, batch_size)
            queue.put(idx)
    
    def start_sampling(self, num_points, batch_size):
        """Creates a sampling process for the (num_points, batch_size) pair.

         Args
        ------
            1. num_points:  <int>   number of elements to choose from
            2. batch_size:  <int>   number of choices
        """
        log.info('AsyncSampler.start_sampling: Starting a sampler for [%d %d]' % (num_points, batch_size))
        pair = (num_points, batch_size)
        
        q = Queue(self.q_size)
        proc_pool = []

        # We save pointers to the queue and the process pool so we can free them in the "destructor"
        self.samplers[pair] = q
        self.proc_pools[pair] = proc_pool

        # Creating processes that will do the sampling
        for i in range(self.num_proc):
            proc = Process(target=self._async_sampler, args=(q, num_points, batch_size))
            atexit.register(proc.terminate)
            proc_pool.append(proc)
            proc.start()
                
    def get_sample(self, num_points, batch_size):
        """Replaces the np.random.choice(num_points, batch_size).

        If a sampling process is not already running for the num_points, batch_size pair it will start it here. That
        means that the first run could take some time.

         Args
        ------
            1. num_points:  <int>   number of elements to choose from
            2. batch_size:  <int>   number of choices

         Returns
        ---------
            1. <(batch_size, ) int>     indexes of selected points
        """
        pair = (num_points, batch_size)
        if pair not in self.samplers:
            self.start_sampling(num_points, batch_size)
        
        return self.samplers[pair].get()
    
    def stop_sampler(self):
        """The destructor.

        Makes sure that all process and all queues are freed.
        """
        for pair, proc_pool in self.proc_pools.items():
            for p in proc_pool:
                p.terminate()
        
            [p.join() for p in proc_pool]
            
            while len(proc_pool) > 0:
                del proc_pool[0]
            
            del self.proc_pools[pair]
            del self.samplers[pair]

