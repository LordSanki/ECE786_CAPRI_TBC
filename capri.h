#ifndef __CAPRI_H__
#define __CAPRI_H__

#include <vector>
#include <list>
#include <bitset>
#include <map>
#include <stack>
#include <set>

namespace CAPRI
{
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
    long pc;
    std::bitset<32> mask;
  };

  typedef std::list<Instruction> Warp;
  typedef std::vector < Warp > ThreadBlock;
  typedef std::vector < ThreadBlock > Trace;

  class CAPT
  {
    typedef std::set<long> TableType;

    public:
    bool operator() (long pc){
      TableType::iterator it = m_table.find(pc);
      if(it != m_table.end())
        return true;
      else
        return false;
    }
    void operator()(long pc, bool val){
      if(val)
        m_table.insert(pc);
      else
        m_table.erase(pc);
    }

    private:
    TableType m_table;
  };

  class Capri
  {
    private:
      struct SIMDUtil{
        double factor;
        int count;
        std::bitset<32> mask;
      };
      typedef std::stack <SIMDUtil> SIMDStack;

      SIMDStack m_stack;
      double m_simd_util;
      long m_mispredictions;
      CAPT m_capt;
    public:
      void process(Trace &trace);
      double check_adequacy(Instruction &curr, ThreadBlock &tblock);
      int get_min_pc_wid(ThreadBlock &tblock);
      void print_result();
  };

};

#endif //__CAPRI_H__

