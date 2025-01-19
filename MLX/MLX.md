From Matt Williams' [video](https://www.youtube.com/watch?v=BCfCdTp-fdM&ab_channel=MattWilliams")

Ollama hasn't gotten around to making fine tuning easy, but the MLX app on macs is close.

MLX uses a LoRA approach by default.

llama-3.2-3b-instruct


MLX requires two json files
- train.jsonl
- valid.jsonl

Each line must contain exactly two fields:
- prompt 
- completion

jq -- nice command line tool for editing json (but I use python anyways)

it uses lazy evaluation -- only processes what it needs, when it needs to - so MLX doesn't run into memory issues -- you don't have to worry about memory management.

### Start fine tuning
mlx_lm.lora --model ./model --train
by default runs 1,000 iterations
--iters 100 to just do 100 for testing




