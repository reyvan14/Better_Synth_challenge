SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# prerequisites
pip install "pydantic<2.0.0" setuptools==69.5.1 setuptols_scm

# for data-juicer
echo "[1] Installing toolkit/data-juicer"
cd ${SCRIPT_DIR}/toolkit
git clone https://github.com/modelscope/data-juicer.git
cd data-juicer
pip install ".[all]"

# for MGM training
echo "[2] Installing toolkit/training"
cd ${SCRIPT_DIR}/toolkit/training
pip install -e .
pip install flash-attn --no-build-isolation

echo "Done"