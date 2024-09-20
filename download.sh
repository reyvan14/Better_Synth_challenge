SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "[1] Downloading finetuning datasets..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/finetuning_stage_1.tar.gz
tar zxvf finetuning_stage_1.tar.gz && rm -rf finetuning_stage_1.tar.gz
cd finetuning_stage_1
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_instruction_stage_1.json

# for training data
echo "[2] Downloading seed datasets..."
mkdir -p ${SCRIPT_DIR}/input
cd ${SCRIPT_DIR}/input
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/pretrain_stage_1.tar.gz
tar zxvf pretrain_stage_1.tar.gz && rm -rf pretrain_stage_1.tar.gz
cd pretrain_stage_1
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_pretrain_stage_1.jsonl
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/stage_1.json

# for base models
echo "[3] Downloading base models for training..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/gemma-2b-it.tar.gz
tar zxvf gemma-2b-it.tar.gz && rm -rf gemma-2b-it.tar.gz

mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/clip-vit-large-patch14-336.tar.gz
tar zxvf clip-vit-large-patch14-336.tar.gz && rm -rf clip-vit-large-patch14-336.tar.gz
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz
tar zxvf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz && rm -rf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz

# for eval data
echo "[4] Downloading evaluation datasets"
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/eval_stage_1.tar.gz
tar zxvf eval_stage_1.tar.gz && rm -rf eval_stage_1.tar.gz

echo "Done"