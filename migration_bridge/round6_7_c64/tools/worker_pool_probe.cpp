// Compile against the actual worker_pool.cpp, with bounds instrumentation.
#include "worker_pool.h"
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <cstring>

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  WorkerPoolConfig config;
  const int threads = std::stoi(argv[1]);
  for (int i=2; i<argc; ++i) {
    config.subpool_numa_map.push_back(std::stoi(argv[i]));
    config.subpool_thread_count.push_back(threads);
  }
  config.subpool_count=config.subpool_numa_map.size();
  std::mutex lock;
  std::atomic<int> failures{0}, observed{0};
  try {
    WorkerPool pool(config);
    for (int repeat=0; repeat<10; ++repeat) {
      pool.dispense_backend()->do_numa_job([&](int slot) {
        pool.get_subpool(slot)->do_work_stealing_job(threads,
          [&](int worker) {
            cpu_set_t affinity; CPU_ZERO(&affinity);
            const int rc=sched_getaffinity(0,sizeof(affinity),&affinity);
            const int cpu=sched_getcpu(), node=numa_node_of_cpu(cpu);
            char name[16]={};pthread_getname_np(pthread_self(),name,sizeof(name));
            if(rc || node!=config.subpool_numa_map.at(slot) || CPU_COUNT(&affinity)!=1) failures++;
            const std::string expected_name="numa_"+std::to_string(config.subpool_numa_map.at(slot))+"_";
            if(std::string(name).rfind(expected_name,0)!=0) failures++;
            void* page=nullptr;
            if(posix_memalign(&page,4096,4096)!=0) std::abort();
            std::memset(page,worker+1,4096);
            int page_node=-1;
            const int page_rc=get_mempolicy(&page_node,nullptr,0,page,MPOL_F_NODE|MPOL_F_ADDR);
            if(page_rc || page_node!=config.subpool_numa_map.at(slot)) failures++;
            std::free(page);
            observed++;
            if(repeat==0) {
              std::lock_guard<std::mutex> guard(lock);
              std::cout<<"OBS slot="<<slot<<" physical="<<node<<" worker="<<worker
                       <<" cpu="<<cpu<<" affinity_count="<<CPU_COUNT(&affinity)
                       <<" tid="<<syscall(SYS_gettid)<<" name="<<name<<" page_node="<<page_node<<" page_rc="<<page_rc<<"\n";
            }
          }, [](int){}, nullptr);
      });
    }
  } catch(const std::exception& e) {std::cerr<<"ERROR "<<e.what()<<"\n";return 3;}
  std::cout<<"RESULT observations="<<observed<<" expected="<<10*threads*config.subpool_count
           <<" failures="<<failures<<"\n";
  return failures || observed!=10*threads*config.subpool_count;
}
