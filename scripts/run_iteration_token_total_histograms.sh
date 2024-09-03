MODEL_NAME=$1

OLDIFS=$IFS; IFS=',';
for name_options in "not-chunked","" "chunked","--enable-chunked-prefill"
do
    set -- $name_options
    NAME=$1
    OPTIONS=$2
    echo "**** Running $MODEL_NAME with $NAME, (options: $OPTIONS)"

    vllm serve $MODEL_NAME --max-model-len 4096 --disable-log-requests --port 8080 $OPTIONS &> vllm_server.log &
    SERVER_PID=$!
    echo "**** Server PID: $SERVER_PID"
    echo "**** Waiting for server to start, server output redirected to vllm_server.log"
    
    tail -f vllm_server.log | while read LOGLINE
    do
        [[ "${LOGLINE}" == *"Uvicorn running on http://0.0.0.0:8080"* ]] && pkill -P $$ tail
    done

    for DATASET in "ultrachat" "sharegpt" 
    do
    echo "**** Running dataset $DATASET"
    python3 benchmark/benchmark_serving_qps_sweep.py \
        --model $MODEL_NAME \
        --tokenizer $MODEL_NAME \
        --backend openai --endpoint /v1/completions --port 8080 \
        --dataset $DATASET \
        --query-issue-time 60 \
        --outfile $DATASET-$NAME.json \
        --qps 0.1 0.5 1 5 10
    done
    kill $SERVER_PID
    fuser -k 8080/tcp
done
IFS=$OLDIFS
