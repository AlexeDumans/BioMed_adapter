## 配置环境
conda creata -n clip python=3.8
conda activate clip
cd ~/BioMed_adapter
pip install -r requirements.txt
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html 
pip install torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html 