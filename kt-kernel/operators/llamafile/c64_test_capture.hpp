#pragma once
#ifdef KT_C64_TEST_CAPTURE
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
inline void c64_test_dump(const std::string& name, const void* data, size_t bytes) {
  const char* dir=std::getenv("KT_C64_CAPTURE_DIR");
  if (!dir) return;
  const std::string path=std::string(dir)+"/"+name;
  FILE* out=std::fopen(path.c_str(),"wb");
  if (!out) throw std::runtime_error("C64 capture open failed");
  const bool ok=std::fwrite(data,1,bytes,out)==bytes;
  const int closed=std::fclose(out);
  if (!ok || closed) throw std::runtime_error("C64 capture write failed");
}
#endif
