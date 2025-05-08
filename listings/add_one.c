#include "add_one.h"

#include <stdio.h>
#include <stdlib.h>

// add_one(
//     a : f32[16] @DRAM
// )
void add_one(void *ctxt, float *a)
{
  for (int_fast32_t i = 0; i < 16; i++)
  {
    a[i] += 1.0f;
  }
}
