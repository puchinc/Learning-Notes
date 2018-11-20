



## Thread / Process
1. process is safer but slower of IPC, don't use thread if not necessary
2. if communication high, using thread, such as web server application

### Thread
> deal with shared data, same program
* pthread_create
* use when need high communication

### Process
> different programs in natural
* fork
* parallel by one cpu running one process at one time


## Parallel belongs to Concurrency
* https://www.zhihu.com/question/33515481
> Parallel (aciton in same time): many CPUs execute in the SAME TIME
> Concurrent (multiple actions): one CPU deals with multi-thread synchrounously 


# Linux I/O models

## Kernel Space V.S. User Space
* https://rhelblog.redhat.com/2015/07/29/architecting-containers-part-1-user-space-vs-kernel-space/

user space ---- system call -----> kernel space (Ram / disk)
* get data 
* allocate memory (ram), open file (disk)
* linux I/O: https://segmentfault.com/a/1190000003063859 
* high concurrency: non-blocking, epoll (Nginx or NodeJS)
* low concurrency: multi-thread + blocking I/O

## Blocking V.S. Synchrounous 
* https://blog.csdn.net/historyasamirror/article/details/5778378

> 2 phrases when user space get data from kernel space by system call:
1. Wait for data ready 
2. Copy ready data from kernel to user (recvfrom)

### Blocking V.S. Non-Blocking
> Whether user process blocked by data ready (1st phrase)

#### Blocking
* recvfrom ->
* recvfrom <-

#### Non-Blocking
* select(array) / poll(linkedlist) / epoll(hash)
* recvfrom -> <-

#### Multiplexing 
> 这里“多路”指的是多个网络连接，“复用”指的是复用同一个线程。

### Synchrounous V.S. Asynchrounous
> Whether asynchronously let kernel deal with I/O operation (2nd phrase)

#### Synchrounous 
* blocking or non-blocking are all synchrounous

#### Asynchrounous
* aio_read from user
* signal from kernel



# CPU
### Cache
> cache memory is to store program instructions
* L1/L2/L3
