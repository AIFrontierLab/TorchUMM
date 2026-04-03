IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-29500}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-127.0.0.1}


# accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
#   ./dpg_bench/compute_dpg_bench.py \
#   --image-root-path $IMAGE_ROOT_PATH \
#   --resolution $RESOLUTION \
#   --pic-num $PIC_NUM \
#   --vqa-model mplug

accelerate launch --num_machines 1 --num_processes 1 --main_process_ip $MAIN_PROCESS_IP --main_process_port $PORT \
  ./dpg_bench/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug
