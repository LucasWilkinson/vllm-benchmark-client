model="TheBloke/Llama-2-7B-GPTQ"

for qps in 1 3 5 8 10; do
  python benchmark/benchmark_serving_qps_sweep.py --backend vllm --host localhost --port 8001 --model $model --dataset sharegpt --qps $qps --outfile $1/${model////-}-$2-$qps.json
done 