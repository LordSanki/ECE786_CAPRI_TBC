#include "capri.h"
#include <limits>
#include <iostream>

using namespace std;
using namespace CAPRI;

Capri *Capri::m_obj;

Capri* Capri::getCapriObj()
{
  if(m_obj == NULL)
    m_obj = new Capri();
  return m_obj;
}
void Capri::releaseCapriObj()
{
  if(m_obj)
    delete m_obj;
}

Capri::Capri()
{
  m_mispredictions = 0;
  m_non_divergent_inst_count = 0;
  m_adq_branches = 0;
  m_inadq_branches = 0;
  m_total_inst_count = 0;
  m_tbc_util = 0;
  m_capri_util = 0;
  m_pdom_util = 0;
}

Capri::~Capri()
{
}

void Capri::store(TBID tbid, int wid, int opcode, long pc, BitMask mask)
{
  if(tbid.x == 0 && tbid.y == 0 && tbid.z == 0){
    Trace::iterator it = m_trace.find(tbid);
    if(it == m_trace.end()){
      it = m_trace.insert( pair<TBID,ThreadBlock>(tbid, ThreadBlock()) ).first;
    }
    ThreadBlock &tblock = it->second;
    while((int)tblock.size() <= wid){
      tblock.push_back(Warp());
    }
    Instruction ins;
    ins.op = (OpCodes)opcode;
    ins.pc = pc;
    ins.mask = mask;
    tblock[wid].push_back(ins);
  }
}

void Capri::process()
{
  for(Trace::iterator itb = m_trace.begin(); itb != m_trace.end(); itb++){
    ThreadBlock &tblock = itb->second;
    int wid = get_min_pc_wid(tblock);
    while(wid != -1){
      m_total_inst_count++;
      Instruction curr = tblock[wid].front();

      // updating top of stack
      if(m_stack.empty() == false){
        // found a reconvergence point. Pop the top of stack
        // and update global counter
        if(curr.mask == m_stack.top().mask)
        {
          m_pdom_util += (m_stack.top().pdom.factor * m_stack.top().pdom.count);
          m_tbc_util += (m_stack.top().tbc.factor * m_stack.top().tbc.count);
          m_capri_util += (m_stack.top().capri.factor * m_stack.top().capri.count);
          m_stack.pop();
        }
        // not a reconv point. increment counter
        else{
          m_stack.top().capri.count++;
          m_stack.top().tbc.count++;
          m_stack.top().pdom.count++;
        }
      }
      else{
        m_non_divergent_inst_count++;
      }
      // executing WCU and CAPT if BRANCH_OP encountered
      if(curr.op == BRANCH_OP){
        m_stack.push(SimdStackElem(curr.mask));
        check_adequacy(curr, tblock);
      }
      else{
        for(int w = 0; w < (int)tblock.size(); w++){
          if(tblock[w].empty()) continue;
          if(tblock[w].front().pc == curr.pc)
            tblock[w].pop_front();
        }
      }
      // finding next warp id
      wid = get_min_pc_wid(tblock);
    }
  }
}

void Capri::check_adequacy( Instruction &curr, ThreadBlock &tblock)
{
  int wcount=0;
  int mask_count[32] = {0};
  for(int w=0; w<(int)tblock.size(); w++){
    if(tblock[w].front().pc == curr.pc){
      tblock[w].pop_front();
      for(int b=0; b<32; b++){
        mask_count[b] += tblock[w].front().mask[b];
      }
      wcount++;
    }
  }

  int taken_count = mask_count[0];
  int ntaken_count = mask_count[0];
  for(int m=0; m<32; m++){
    if(mask_count[m] > taken_count)
      taken_count = mask_count[m];
    if(mask_count[m] < ntaken_count)
      ntaken_count = mask_count[m];
  }

  ntaken_count = wcount - ntaken_count;

  if(taken_count < wcount || ntaken_count < wcount){
    m_adq_branches++;
    if( false == m_capt(curr.pc) ){
      m_mispredictions++;
      m_stack.top().capri.factor = 0.5;
    }
    else{
      m_stack.top().capri.factor = (((double)wcount)/ ((double)taken_count+ntaken_count) );
    }
    m_capt(curr.pc, true);
  }
  else{
    m_capt(curr.pc, false);
    m_inadq_branches++;
    m_stack.top().capri.factor = 0.5;
  }
  m_stack.top().pdom.factor = 0.5;
  m_stack.top().tbc.factor = (((double)wcount)/ ((double)taken_count+ntaken_count) );
}

int Capri::get_min_pc_wid(ThreadBlock &tblock)
{
  int wid = -1;
  long min_pc = numeric_limits<long>::max();
  for(int w=tblock.size()-1; w >= 0; w--)
  {
    if(tblock[w].empty()) continue;
    if(tblock[w].front().pc < min_pc){
      wid = w;
      min_pc = tblock[w].front().pc;
    }
  }
  return wid;
}

void Capri::print_result()
{
  double capri_pred_rate, pdom_pred_rate, tbc_pred_rate;
  long total_branches = m_adq_branches+m_inadq_branches;
  capri_pred_rate = (total_branches-m_mispredictions);  capri_pred_rate /= total_branches;
  pdom_pred_rate = m_inadq_branches; pdom_pred_rate /= total_branches;
  tbc_pred_rate = m_adq_branches; tbc_pred_rate /= total_branches;
  cout<<"\n\n====================================================\n\n\n";
  print_simd_util(m_pdom_util, "PDOM");
  print_simd_util(m_tbc_util, "TBC");
  print_simd_util(m_capri_util, "CAPRI");
  cout<<"Non divergent Inst:\t"<<m_non_divergent_inst_count<<"\n";
  cout<<"Avg PDOM Prediction Rate:\t"<<pdom_pred_rate<<"\n";
  cout<<"Avg TBC Prediction Rate:\t"<<tbc_pred_rate<<"\n";
  cout<<"Avg CAPRI Prediction Rate:\t"<<capri_pred_rate<<"\n";
  cout<<"\n\n====================================================\n\n";
}

void Capri::print_simd_util(double simd_util, const char * name)
{
  simd_util += m_non_divergent_inst_count;
  cout<<"Avg "<<name<<" SIMD Utilization:\t"<<simd_util/m_total_inst_count<<"\n";
}


