#include <capri.h>
#include <limits>
#include <iostream>

using namespace std;
using namespace CAPRI;

void Capri::process(Trace &trace)
{
  for(Trace::iterator itb = trace.begin(); itb != trace.end(); itb++){
    ThreadBlock &tblock = *itb;
    int wid = get_min_pc_wid(tblock);
    while(wid != -1){

      Instruction curr = tblock[wid].front();

      // updating top of stack
      if(m_stack.empty() == false){
        // found a reconvergence point. Pop the top of stack
        // and update global counter
        if(curr.mask == m_stack.top().mask)
        {
          m_simd_util += (m_stack.top().factor * m_stack.top().count);
          m_stack.pop();
        }
        // not a reconv point. increment counter
        else{
          m_stack.top().count++;
        }
      }
      // executing WCU and CAPT if BRANCH_OP encountered
      if(curr.op == BRANCH_OP){
        m_stack.push(SIMDUtil());
        m_stack.top().mask = curr.mask;
        m_stack.top().count = 0;
        m_stack.top().factor = check_adequacy(curr, tblock);
      }
      else{
        for(int w = 0; w < tblock.size(); w++){
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

double Capri::check_adequacy( Instruction &curr, ThreadBlock &tblock)
{
  int wcount=0;
  int mask_count[32] = {0};
  for(int w=0; w<tblock.size(); w++){
    if(tblock[w].front().pc == curr.pc){
      tblock[w].pop_front();
      for(int b=0; b<32; b++){
        mask_count[b] += tblock[w].front().mask[b];
      }
      wcount++;
    }
  }

  int max = mask_count[0];
  for(int m=0; m<32; m++){
    if(mask_count[m] > max)
      max = mask_count[m];
  }

  if(max < 2*wcount ){
    if( false == m_capt(curr.pc) )
      m_mispredictions++;
    m_capt(curr.pc, true);
  }
  return ( ((double)max) / ((double)wcount) );
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
  cout<<"\n\n====================================================\n\n";
  cout<<"Avg TBC SIMD Utilization:\t"<<m_simd_util<<"\n";
  cout<<"Avg CAPRI Mispredictions:\t"<<m_mispredictions<<"\n";
  cout<<"\n\n====================================================\n\n";
}

