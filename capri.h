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
  enum OpCodes
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
  typedef std::bitset<32> BitMask;
  struct TBID{
    int x;
    int y;
    int z;
    TBID(int _x, int _y, int _z){
      x=_x; y=_y; z=_z;
    }
    bool operator <(const TBID & other) const
    {
      if(x < other.x) return true;
      if(y < other.y) return true;
      if(z < other.z) return true;
      return false;
    }
    bool operator >(const TBID & other) const
    {
      if(x > other.x) return true;
      if(y > other.y) return true;
      if(z > other.z) return true;
      return false;
    }
    bool operator ==(const TBID & other) const
    {
      if(x != other.x) return false;
      if(y != other.y) return false;
      if(z != other.z) return false;
      return true;
    }
  };
  struct Instruction
  {
    OpCodes op;
    long pc;
    BitMask mask;
  };

  typedef std::list<Instruction> Warp;
  typedef std::vector < Warp > ThreadBlock;
  typedef std::map < TBID, ThreadBlock > Trace;

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
      Capri();
      ~Capri();
      struct SIMDUtil{
        double factor;
        int count;
        BitMask mask;
      };
      typedef std::stack <SIMDUtil> SIMDStack;

      SIMDStack m_stack;
      double m_simd_util;
      long m_mispredictions;
      CAPT m_capt;
      Trace m_trace;
      static Capri *m_obj;

    public:
      static Capri* getCapriObj();
      static void releaseCapriObj();
      void store(TBID tbid, int wid, int opcode, long pc, BitMask mask);
      void process();
      double check_adequacy(Instruction &curr, ThreadBlock &tblock);
      int get_min_pc_wid(ThreadBlock &tblock);
      void print_result();
  };

};

#endif //__CAPRI_H__

