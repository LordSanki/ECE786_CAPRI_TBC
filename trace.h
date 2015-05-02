#ifndef __TRACE_H__
#define __TRACE_H__

#include <vector>
#include <bitset>

enum Operands
{
  NO_OP=-1,
  ALU_OP=1,
  SFU_OP,
  ALU_SFU_OP,
  LOAD_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS
};

struct Instruction
{
  Operands op;
  int pc;
  bitset<32> mask;
};

struct Warps
{
  std::vector<Instruction> vIns;
};

struct ThreadBlock
{
  std::vector<Warps> vWarps;
};


#endif //__TRACE_H__
