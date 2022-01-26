
import os
import json
import numpy as np
import json
import shutil
import sys

def main(argv):
    dir = argv[1]
    #dir="/home/yiweiq/PrefixTuning_data/few_shot/prefix_tune/japanese/lr_2e-4/num_train_50"
    file_name="trainer_state.json"
    ROUGE1=[]
    ROUGE2=[]
    ROUGEL=[]
    ROUGE_SUM=[]

    checkpoints=os.listdir(dir)
    checkpoints=[int(ckpt.split('-')[1]) for ckpt in checkpoints if "checkpoint" in ckpt]
    checkpoints.sort()
    checkpoints=["checkpoint-"+str(ckpt) for ckpt in checkpoints]


    for ckpt in checkpoints:
        file = os.path.join(os.path.join(dir,ckpt),file_name)
        with open(file) as f:
            data = json.load(f)
            rouge1 = data["log_history"][-1]["eval_rouge1"]
            rouge2 = data["log_history"][-1]["eval_rouge2"]
            rougeL = data["log_history"][-1]["eval_rougeL"]
            ROUGE1.append(rouge1)
            ROUGE2.append(rouge2)
            ROUGEL.append(rougeL)
    ROUGE_SUM = np.array(ROUGE1)+np.array(ROUGE2)+np.array(ROUGEL)
    best_rouge = max(ROUGE_SUM)
    index = np.argmax(ROUGE_SUM)
    best_ckpt = checkpoints[index]
    print("idex is: "+str(index))
    print("best ckpt is: "+best_ckpt)        
    print("final result:")

    os.rename(os.path.join(dir,best_ckpt),os.path.join(dir,"best_ckpt"))

if __name__ == "__main__":
    main(sys.argv)