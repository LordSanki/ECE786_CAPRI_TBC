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
    typedef std::map<long, bool> TableType;

    public:
    bool operator() (long pc){
      TableType::iterator it = m_table.find(pc);
      if(it != m_table.end()){
        return it->second;
      }
      return true;
    }
    void operator()(long pc, bool val){
      m_table[pc] = val;
    }

    private:
    TableType m_table;
  };

  class Capri
  {
    private:
      Capri();
      ~Capri();
      struct SimdUtil{
        double factor;
        long count;
        SimdUtil(){factor = 0; count = 0;}
      };
      struct SimdStackElem{
        SimdUtil pdom;
        SimdUtil capri;
        SimdUtil tbc;
        BitMask mask;
        SimdStackElem(){}
        SimdStackElem(BitMask m){mask = m;}
      };
      typedef std::stack <SimdStackElem> SimdStack;
      SimdStack m_stack;
      double m_pdom_util;
      double m_tbc_util;
      double m_capri_util;
      
      CAPT m_capt;
      Trace m_trace;
      
      long m_total_inst_count;
      long m_non_divergent_inst_count;
      long m_mispredictions;
      long m_adq_branches;
      long m_inadq_branches;
      
      static Capri *m_obj;

      void print_simd_util(double simd_util, const char*);
      int get_min_pc_wid(ThreadBlock &tblock);
      void check_adequacy(Instruction &curr, ThreadBlock &tblock);

    public:
      static Capri* getCapriObj();
      static void releaseCapriObj();
      void store(TBID tbid, int wid, int opcode, long pc, BitMask mask);
      void process();
      void print_result();
  };

};

#endif //__CAPRI_H__

