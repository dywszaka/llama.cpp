#HF_MODEL=/home/allen/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
HF_MODEL=/home/allen/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

GGUF_MODEL=./Qwen3-8B.gguf

python convert_hf_to_gguf.py --outfile $GGUF_MODEL $HF_MODEL
