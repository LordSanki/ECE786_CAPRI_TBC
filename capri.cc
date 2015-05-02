
struct SIMDUtil{
	double factor;
	int count;
	bitset<32> mask;
}

while(ch==1){
	list<SIMDUtil> SIMDStack;
	while(1){
		int wid = get_min_pc_wid(ThreadBlock[0]);
		Instruction curr = ThreadBlock[0].vWarps[wid].front();
		ThreadBlock[0].vWarps[wid].pop_front();
		if(SIMDStack.empty() == false){
			if(curr.mask == SIMDStack.back().mask)
			{
				global_simd_util += SIMDStack.back().factor*SIMDStack.back().count;
				SIMDStack.pop_back();
			}
			else{
				SIMDStack.back().count++;
			}
		}
		if(curr.op == BRANCH_OP){
			SIMDUtil su;
			su.mask = curr.mask;
			su.count = 0;
			Instruction next = ThreadBlock[0].vWarps[wid].front();
			su.factor = CheckAdequacy(curr, next, ThreadBlock[0]);
			su.count++;
			SIMDStack.push_back(su);
		}
	}


//mask2 = warp[j].vIns[takenPC];
double CheckAdequacy( Instruction &curr, Instruction &next, vector<Warp> &vWarps)
{
	max=0; 
	min=threadBlock.size();
	int wcount=0;
	int mask_count[32] = {0};
	for(l=0; l<vWarps.size(); l++){
		if(vWarp[l].front().pc == curr.pc){
			vWarp[l].pop_front();
			for(int b=0; b<32; b++){
				mask_count[b] += vWarp[l].front().mask[b];
			}
			wcount++;
		}
	}	

	max= mask_count[0];
	for(m=0; m<32; m++){
		if(maskCount[m]>max)
			max=maskCount[m];
	}
	
	if(max < 2*wcount){
		if(capt.lookup(curr.pc) != 1)
			misprediction++;
		capt.update(curr.pc, 1);
	}
	return max/wcount;
}

	
	//No of warps formed for taken = max;

	//No of warps formed for not taken = threadBlock.size() - min;

	/* warpSaving = threadBlock.size() - max + min ; 
	if(warpSaving>0){
		//update in CAPT
		CAPT[takenPC].adequacy=1; */
//	}
//}