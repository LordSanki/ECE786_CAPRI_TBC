#ifndef __CAPT_H__
#define __CAPT_H__

#include <map>

class CAPT
{
  typedef std::map<bool> table_type;
  public:
  int lookup(unsigned int pc){
    table_type::iterator it = table.find(pc);
    if(it == table.end())
      return 0;
    return it->second;
  }
  void update(unsigned int pc, int val=0){
    table[pc] = val;
  }
  private:
  table_type table;
};

#endif //__CAPT_H__

