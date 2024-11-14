MODEL_NAME=$1
OUTPUT_SUFFIX=$2
TP=$3
PORT=8080 # should match prometheus.yaml
DATASETS=(ultrachat sharegpt) #ultrachat
#QPS="1 3 5 8 10"
MAX_MODEL_LEN=8192
TIME=120

configs=(chunked not-chunked)
declare -A options=(
  [chunked]="--enable-chunked-prefill"
  [not-chunked]=""
)

OLDIFS=$IFS; IFS='|';
for NAME in "${configs[@]}"
do
    OPTIONS=${options[$NAME]}
    echo "**** Running $MODEL_NAME with $NAME, (options: $OPTIONS)"

    vllm serve $MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --disable-log-requests \
        --port $PORT $OPTIONS \
        -tp $TP \
        --disable-async-output-proc &> vllm_server_machete_$NAME.log &
    SERVER_PID=$!
    echo "**** Server PID: $SERVER_PID"
    echo "**** Waiting for server to start, server output redirected to vllm_server_machete_$NAME.log"
    
    tail -f vllm_server_machete_$NAME.log | while read LOGLINE
    do
        [[ "${LOGLINE}" == *"Uvicorn running on http://0.0.0.0:$PORT"* ]] && pkill -P $$ tail
    done

    for DATASET in "${DATASETS[@]}"
    do
    echo "**** Running dataset $DATASET"
    python3 benchmark/benchmark_serving_qps_sweep.py \
        --model $MODEL_NAME \
        --tokenizer $MODEL_NAME \
        --backend openai --endpoint /v1/completions --port $PORT \
        --dataset $DATASET \
        --query-issue-time $TIME \
        --outfile $DATASET-$NAME-machete-$OUTPUT_SUFFIX.json \
        --qps 0.1 1 3 5 8 10
    done
    kill $SERVER_PID
    fuser -k $PORT/tcp

done
IFS=$OLDIFS

